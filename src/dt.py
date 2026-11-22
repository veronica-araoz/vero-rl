"""Decision Transformer para recomendación (Chen et al., 2021).

Secuencia por episodio: R̂_0, s_0, a_0, R̂_1, s_1, a_1, ..., R̂_{T-1}, s_{T-1}, a_{T-1}

- R̂_t: retorno pendiente (suma de ratings desde t hasta el final del episodio), escalado /100.
- s_t: grupo (o "desconocido") + rating del paso anterior r_{t-1} ("ninguno" en t = 0).
       No incluye r_t, que se revela después de mostrar a_t.
- a_t: ítem mostrado.

a_t se predice desde la salida del token s_t, que por la máscara causal ve
R̂_≤t, s_≤t y a_<t, pero no a_t.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import N_GROUPS, UNKNOWN_GROUP, hide_groups, iterate_minibatches, set_seed

RTG_SCALE = 100.0
NO_PREV_RATING = 0  # índice del "sin rating previo"; los ratings 1..5 usan índices 1..5


def prev_ratings_index(ratings: torch.Tensor) -> torch.Tensor:
    """(B, T) ratings -> (B, T) índice del rating anterior: 0 en t = 0, luego round(r_{t-1})."""
    idx = ratings.round().clamp(1, 5).long()
    return torch.cat([torch.full_like(idx[:, :1], NO_PREV_RATING), idx[:, :-1]], dim=1)


class DecisionTransformer(nn.Module):
    def __init__(self, n_items: int, horizon: int = 20, d_model: int = 128, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_items, self.horizon = n_items, horizon
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_group = nn.Embedding(N_GROUPS + 1, d_model)       # +1: desconocido
        self.embed_prev_rating = nn.Embedding(6, d_model)            # 0 = ninguno, 1..5
        self.embed_item = nn.Embedding(n_items + 1, d_model)         # +1: relleno para a_t aún no elegido
        self.embed_step = nn.Embedding(horizon, d_model)
        self.embed_type = nn.Embedding(3, d_model)                   # 0 = R̂, 1 = s, 2 = a
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln_out = nn.LayerNorm(d_model)
        self.out_bias = nn.Parameter(torch.zeros(n_items))
        # La salida comparte pesos con el embedding de ítems (sin la fila de relleno).
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Inicialización chica (estilo GPT): con pesos compartidos entre embedding y salida,
        # embeddings N(0, 1) dan logits enormes al inicio y la CE arranca muy por encima de log(n_items).
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, rtg: torch.Tensor, items: torch.Tensor, prev_rating: torch.Tensor,
                groups: torch.Tensor) -> torch.Tensor:
        """rtg, items, prev_rating: (B, T); groups: (B,). Devuelve logits (B, T, n_items)."""
        B, T = items.shape
        steps = self.embed_step(torch.arange(T, device=items.device))[None]          # (1, T, d)
        tok_r = self.embed_rtg((rtg / RTG_SCALE).unsqueeze(-1)) + self.embed_type.weight[0]
        tok_s = (self.embed_group(groups)[:, None, :] + self.embed_prev_rating(prev_rating)
                 + self.embed_type.weight[1])
        tok_a = self.embed_item(items) + self.embed_type.weight[2]
        x = torch.stack([tok_r + steps, tok_s + steps, tok_a + steps], dim=2).reshape(B, 3 * T, -1)
        mask = torch.triu(torch.full((3 * T, 3 * T), float("-inf"), device=items.device), diagonal=1)
        h = self.transformer(self.drop(x), mask=mask, is_causal=True)
        h_state = self.ln_out(h[:, 1::3])                                              # salidas de s_t
        return h_state @ self.embed_item.weight[: self.n_items].T + self.out_bias


def seen_before_mask(items: torch.Tensor, n_items: int) -> torch.Tensor:
    """(B, T, n_items): True si el ítem ya se mostró antes del paso t en el episodio."""
    onehot = F.one_hot(items, n_items).cumsum(dim=1)
    seen = torch.zeros_like(onehot, dtype=torch.bool)
    seen[:, 1:] = onehot[:, :-1] > 0
    return seen


def dt_loss(model: DecisionTransformer, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    items = batch["items"]
    logits = model(batch["rtg"], items, prev_ratings_index(batch["ratings"]), batch["groups"])
    logits = logits.masked_fill(seen_before_mask(items, model.n_items), float("-inf"))
    return F.cross_entropy(logits.reshape(-1, model.n_items), items.reshape(-1))


# ----------------------------------------------------------------------------- entrenamiento

@dataclass
class DTConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 256
    epochs: int = 6
    p_hidden_group: float = 0.3
    grad_clip: float = 1.0
    seed: int = 0


def train_dt(train_data: dict, val_data: dict, n_items: int, cfg: DTConfig, log_every: int = 20):
    """Entrena y devuelve (modelo con el mejor checkpoint en val, historia de curvas)."""
    set_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = DecisionTransformer(n_items, train_data["items"].shape[1], cfg.d_model, cfg.n_layers,
                                cfg.n_heads, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_batches = -(-train_data["items"].shape[0] // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.epochs * n_batches)
    history = {"step": [], "train_loss": [], "epoch": [], "val_loss": [], "config": asdict(cfg)}
    best, best_state, step, t0 = float("inf"), None, 0, time.time()
    for epoch in range(cfg.epochs):
        model.train()
        acc = []
        for batch in iterate_minibatches(train_data, cfg.batch_size, True, gen):
            batch = {**batch, "groups": hide_groups(batch["groups"], cfg.p_hidden_group, gen)}
            loss = dt_loss(model, batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            acc.append(loss.item())
            if step % log_every == 0:
                history["step"].append(step)
                history["train_loss"].append(float(np.mean(acc)))
                acc = []
        val_loss = evaluate_dt_loss(model, val_data, cfg.batch_size)
        history["epoch"].append(epoch + 1)
        history["val_loss"].append(val_loss)
        print(f"[DT seed {cfg.seed}] época {epoch + 1}/{cfg.epochs} val CE {val_loss:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    history["best_val_loss"] = best
    return model.eval(), history


@torch.no_grad()
def evaluate_dt_loss(model, data, batch_size=1024, p_hidden_group: float = 0.0) -> float:
    """CE media en validación (grupo visible). Referencia uniforme: log(n_items)."""
    model.eval()
    total, n = 0.0, 0
    for batch in iterate_minibatches(data, batch_size, False):
        k = batch["items"].numel()
        total += dt_loss(model, batch).item() * k
        n += k
    return total / n


# ----------------------------------------------------------------------------- política

class DTPolicy:
    """Política para el protocolo. Interfaz por lotes (ver utils.BatchPolicy) y simple."""

    def __init__(self, model: DecisionTransformer, target_return: float):
        self.model = model.eval()
        self.target = float(target_return)
        self.groups = None

    def reset_batch(self, groups, horizon):
        self.groups = torch.tensor([UNKNOWN_GROUP if g is None else g for g in groups], dtype=torch.long)

    @torch.no_grad()
    def scores_batch(self, items_seen: np.ndarray, ratings_seen: np.ndarray) -> np.ndarray:
        B, t = items_seen.shape
        ratings = torch.as_tensor(ratings_seen, dtype=torch.float32)
        # R̂_k = objetivo − ratings ya revelados antes de k, para k = 0..t
        cum = torch.cat([torch.zeros(B, 1), ratings.cumsum(1)], dim=1)                 # (B, t+1)
        rtg = self.target - cum
        pad = torch.full((B, 1), self.model.n_items, dtype=torch.long)                   # a_t aún no existe
        items = torch.cat([torch.as_tensor(items_seen, dtype=torch.long), pad], dim=1)
        ratings_full = torch.cat([ratings, torch.zeros(B, 1)], dim=1)                    # r_t no se usa
        logits = self.model(rtg, items, prev_ratings_index(ratings_full), self.groups)[:, t]
        return logits.numpy()

    # interfaz simple del protocolo
    def reset(self, group, horizon):
        self.reset_batch([group], horizon)

    def scores(self, items_seen, ratings_seen):
        return self.scores_batch(np.asarray(items_seen)[None, :], np.asarray(ratings_seen)[None, :])[0]
