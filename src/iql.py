"""Implicit Q-Learning (Kostrikov, Nair y Levine, 2022) para recomendación.

Tres redes, cada una con su propio codificador del historial (GRU):
- V(s): regresión de expectil sobre Q̄(s, a) de las acciones registradas.
- Q(s, a): diferencias temporales con r + γ·V(s'); en el último paso, sólo r.
- π(a | s): regresión ponderada por ventaja (AWR), pesos exp(β·(Q̄ − V)) truncados.
Q̄ es la red objetivo (copia de Q actualizada con promedio de Polyak).

Estado s_t = (grupo o "desconocido", pares (a_k, r_k) para k < t, paso t).
Recompensa centrada: (r − 3) / 2, en [−1, 1].
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dt import seen_before_mask
from .utils import N_GROUPS, UNKNOWN_GROUP, hide_groups, iterate_minibatches, set_seed


def center_reward(r: torch.Tensor) -> torch.Tensor:
    return (r - 3.0) / 2.0


class StateEncoder(nn.Module):
    """Devuelve T + 1 estados h_0..h_T a partir de (grupo, historial)."""

    def __init__(self, n_items: int, horizon: int, d: int):
        super().__init__()
        self.n_items = n_items
        self.embed_item = nn.Embedding(n_items + 1, d)       # n_items = token de inicio
        self.embed_rating = nn.Embedding(6, d)               # 0 = ninguno, 1..5
        self.embed_group = nn.Embedding(N_GROUPS + 1, d)
        self.embed_step = nn.Embedding(horizon + 1, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, items: torch.Tensor, ratings: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """items, ratings: (B, t) observados; groups: (B,). Devuelve (B, t + 1, d)."""
        B, t = items.shape
        start_i = torch.full((B, 1), self.n_items, dtype=torch.long, device=items.device)
        start_r = torch.zeros((B, 1), dtype=torch.long, device=items.device)
        it = torch.cat([start_i, items], 1)
        rt = torch.cat([start_r, ratings.round().clamp(1, 5).long()], 1)
        x = self.embed_item(it) + self.embed_rating(rt) + self.embed_group(groups)[:, None]
        h, _ = self.gru(x)
        steps = self.embed_step(torch.arange(t + 1, device=items.device))[None]
        return self.norm(h + steps + self.embed_group(groups)[:, None])


class Head(nn.Module):
    def __init__(self, d: int, out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, out))

    def forward(self, h):
        return self.net(h)


class Net(nn.Module):
    """Codificador + cabeza. Se usa para Q (salida n_items), V (salida 1) y π (logits n_items)."""

    def __init__(self, n_items, horizon, d, out):
        super().__init__()
        self.enc = StateEncoder(n_items, horizon, d)
        self.head = Head(d, out)

    def forward(self, items, ratings, groups):
        return self.head(self.enc(items, ratings, groups))           # (B, t + 1, out)


class IQL(nn.Module):
    def __init__(self, n_items: int, horizon: int = 20, d: int = 128):
        super().__init__()
        self.n_items = n_items
        self.q = Net(n_items, horizon, d, n_items)
        self.v = Net(n_items, horizon, d, 1)
        self.pi = Net(n_items, horizon, d, n_items)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)

    @torch.no_grad()
    def update_target(self, polyak: float):
        for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
            p_t.mul_(1 - polyak).add_(polyak * p)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """diff = Q_obj − V. Con τ > 0,5 pesa más quedarse por debajo (diff > 0)."""
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


def iql_losses(model: IQL, batch, gamma: float, tau: float, beta: float, max_weight: float = 100.0):
    items, ratings, groups = batch["items"], batch["ratings"], batch["groups"]
    T = items.shape[1]
    r = center_reward(ratings)
    not_done = torch.ones_like(r)
    not_done[:, -1] = 0.0                                              # último paso: terminal

    with torch.no_grad():
        q_obj = model.q_target(items, ratings, groups)[:, :T].gather(-1, items.unsqueeze(-1)).squeeze(-1)

    v_all = model.v(items, ratings, groups).squeeze(-1)               # (B, T + 1): V(s_0..s_T)
    loss_v = expectile_loss(q_obj - v_all[:, :T], tau)

    q = model.q(items, ratings, groups)[:, :T].gather(-1, items.unsqueeze(-1)).squeeze(-1)
    td_target = r + gamma * not_done * v_all[:, 1:].detach()
    loss_q = F.mse_loss(q, td_target)

    with torch.no_grad():
        adv = q_obj - v_all[:, :T]
        raw_w = torch.exp(beta * adv)
        weights = raw_w.clamp(max=max_weight)
    logits = model.pi(items, ratings, groups)[:, :T]
    logits = logits.masked_fill(seen_before_mask(items, model.n_items), float("-inf"))
    nll = F.cross_entropy(logits.reshape(-1, model.n_items), items.reshape(-1), reduction="none")
    loss_pi = (weights.reshape(-1) * nll).mean()

    stats = {"w_mean": weights.mean().item(), "w_clipped": (raw_w >= max_weight).float().mean().item(),
             "adv_mean": adv.mean().item(), "adv_std": adv.std().item()}
    return {"v": loss_v, "q": loss_q, "pi": loss_pi}, stats


# ----------------------------------------------------------------------------- entrenamiento

@dataclass
class IQLConfig:
    gamma: float = 0.9
    tau: float = 0.7
    beta: float = 3.0
    max_weight: float = 100.0
    d: int = 128
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 5
    polyak: float = 0.005
    p_hidden_group: float = 0.3
    grad_clip: float = 1.0
    seed: int = 0


def train_iql(train_data: dict, n_items: int, cfg: IQLConfig, log_every: int = 20):
    set_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = IQL(n_items, train_data["items"].shape[1], cfg.d)
    opts = {k: torch.optim.Adam(getattr(model, k).parameters(), lr=cfg.lr) for k in ("q", "v", "pi")}
    keys = ["loss_v", "loss_q", "loss_pi", "w_mean", "w_clipped", "adv_mean", "adv_std"]
    history = {"step": [], **{k: [] for k in keys}, "config": asdict(cfg)}
    acc = {k: [] for k in keys}
    step, t0 = 0, time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for batch in iterate_minibatches(train_data, cfg.batch_size, True, gen):
            batch = {**batch, "groups": hide_groups(batch["groups"], cfg.p_hidden_group, gen)}
            losses, stats = iql_losses(model, batch, cfg.gamma, cfg.tau, cfg.beta, cfg.max_weight)
            for o in opts.values():
                o.zero_grad()
            # Cada pérdida sólo llega a su red (V(s') y los pesos van sin gradiente).
            (losses["v"] + losses["q"] + losses["pi"]).backward()
            for k, o in opts.items():
                nn.utils.clip_grad_norm_(getattr(model, k).parameters(), cfg.grad_clip)
                o.step()
            model.update_target(cfg.polyak)
            step += 1
            for k in ("v", "q", "pi"):
                acc[f"loss_{k}"].append(losses[k].item())
            for k, v in stats.items():
                acc[k].append(v)
            if step % log_every == 0:
                history["step"].append(step)
                for k in keys:
                    history[k].append(float(np.mean(acc[k])))
                    acc[k] = []
        print(f"[IQL γ={cfg.gamma} τ={cfg.tau} β={cfg.beta} seed {cfg.seed}] época {epoch + 1}/{cfg.epochs} "
              f"Lv {history['loss_v'][-1]:.4f} Lq {history['loss_q'][-1]:.4f} Lπ {history['loss_pi'][-1]:.3f} "
              f"w̄ {history['w_mean'][-1]:.2f} ({time.time() - t0:.0f}s)", flush=True)
    return model.eval(), history


# ----------------------------------------------------------------------------- política

class IQLPolicy:
    """mode="pi": ordena por las logits de π (AWR). mode="q": ordena por Q (máximo de Q)."""

    def __init__(self, model: IQL, mode: str = "pi"):
        assert mode in ("pi", "q")
        self.model, self.mode = model.eval(), mode
        self.groups = None

    def reset_batch(self, groups, horizon):
        self.groups = torch.tensor([UNKNOWN_GROUP if g is None else g for g in groups], dtype=torch.long)

    @torch.no_grad()
    def scores_batch(self, items_seen, ratings_seen):
        items = torch.as_tensor(items_seen, dtype=torch.long)
        ratings = torch.as_tensor(ratings_seen, dtype=torch.float32)
        net = self.model.pi if self.mode == "pi" else self.model.q
        return net(items, ratings, self.groups)[:, -1].numpy()      # salida del estado actual s_t

    def reset(self, group, horizon):
        self.reset_batch([group], horizon)

    def scores(self, items_seen, ratings_seen):
        return self.scores_batch(np.asarray(items_seen)[None, :], np.asarray(ratings_seen)[None, :])[0]
