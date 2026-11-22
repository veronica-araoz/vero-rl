"""Utilidades comunes: rutas, semillas, episodios en tensores y evaluación por lotes.

La evaluación por lotes (`evaluate_batched`) reproduce exactamente
`protocolo.evaluar.run_session` (mismas semillas, mismo warm-up, mismo `choose`
y mismas métricas), pero pide los puntajes de todas las sesiones de un paso en
una sola llamada al modelo. Se verifica contra la función oficial en
`tests/test_modelos.py`.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protocolo.datos import RELEVANCE_THRESHOLD, Episodes, User  # noqa: E402
from protocolo.metricas import normalized_score, ranking_metrics, session_bounds  # noqa: E402
from protocolo.simulador import Session, SessionConfig, choose  # noqa: E402

N_GROUPS = 8
UNKNOWN_GROUP = N_GROUPS  # índice del token "grupo desconocido"
OUTPUTS = ROOT / "outputs"


# ----------------------------------------------------------------------------- semillas

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------- episodios

def episodes_to_tensors(ep: Episodes) -> dict[str, torch.Tensor]:
    """Episodios (N, T) a tensores. Incluye el retorno pendiente R̂_t."""
    return {
        "items": torch.as_tensor(ep.items, dtype=torch.long),
        "ratings": torch.as_tensor(ep.ratings, dtype=torch.float32),
        "rtg": torch.as_tensor(ep.returns_to_go().copy(), dtype=torch.float32),
        "groups": torch.as_tensor(ep.groups, dtype=torch.long),
    }


def hide_groups(groups: torch.Tensor, p_hidden: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Reemplaza el grupo por UNKNOWN_GROUP en una fracción p_hidden de los episodios.

    Así el modelo aprende qué hacer cuando el grupo está oculto al evaluar.
    """
    if p_hidden <= 0:
        return groups
    hide = torch.rand(groups.shape, generator=generator) < p_hidden
    return torch.where(hide, torch.full_like(groups, UNKNOWN_GROUP), groups)


def iterate_minibatches(data: dict[str, torch.Tensor], batch_size: int, shuffle: bool,
                        generator: torch.Generator | None = None):
    n = data["items"].shape[0]
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield {k: v[idx] for k, v in data.items()}


# ----------------------------------------------------------------------------- evaluación

class BatchPolicy:
    """Interfaz por lotes. `groups[i]` es None si el grupo está oculto.

    `scores_batch(items_seen, ratings_seen)` recibe arrays (B, t) y devuelve (B, n_items).
    """

    def reset_batch(self, groups: list[int | None], horizon: int) -> None: ...

    def scores_batch(self, items_seen: np.ndarray, ratings_seen: np.ndarray) -> np.ndarray: ...


class SinglePolicyAdapter:
    """Adapta una BatchPolicy a la interfaz `Policy` del protocolo (una sesión a la vez)."""

    def __init__(self, batch_policy: BatchPolicy):
        self.p = batch_policy

    def reset(self, group, horizon):
        self.p.reset_batch([group], horizon)

    def scores(self, items_seen, ratings_seen):
        return self.p.scores_batch(items_seen[None, :], ratings_seen[None, :])[0]


def evaluate_batched(policy: BatchPolicy, users: list[User], n_items: int, config: SessionConfig,
                     seeds: Iterable[int] = (0, 1, 2), chunk: int = 4096) -> pd.DataFrame:
    """Equivalente a `protocolo.evaluar.evaluate`, con el modelo llamado por lotes."""
    jobs = [(seed, index, user) for seed in seeds for index, user in enumerate(users)]
    rows = []
    for start in range(0, len(jobs), chunk):
        part = jobs[start:start + chunk]
        rngs = [np.random.default_rng([seed, index]) for seed, index, _ in part]
        sessions = [Session(user, n_items, config, rng) for (_, _, user), rng in zip(part, rngs)]
        policy.reset_batch([s.visible_group for s in sessions], config.horizon)
        steps = config.controlled_steps
        bounds = [session_bounds(s._available_hidden_ratings(), steps) for s in sessions]
        controlled = [[] for _ in sessions]
        first = [None] * len(sessions)
        while not sessions[0].done:
            obs = [s.observed() for s in sessions]
            items_seen = np.stack([o[0] for o in obs])
            ratings_seen = np.stack([o[1] for o in obs])
            all_scores = np.asarray(policy.scores_batch(items_seen, ratings_seen), dtype=np.float64)
            for i, s in enumerate(sessions):
                mask = s.available_mask()
                if first[i] is None:
                    first[i] = ranking_metrics(all_scores[i], s._hidden_ratings(), mask, k=10)
                controlled[i].append(s.step(choose(all_scores[i], mask, rngs[i])))
        for i, ((seed, index, user), s) in enumerate(zip(part, sessions)):
            ret = float(np.sum(controlled[i]))
            rand_exp, oracle = bounds[i]
            rows.append({
                "user_id": user.user_id, "group": user.group, "seed": seed,
                "return": ret, "full_return": float(np.sum(s.ratings)),
                "mean_rating": ret / steps,
                "hit_rate": float(np.mean(np.array(controlled[i]) >= RELEVANCE_THRESHOLD)),
                "normalized": normalized_score(ret, rand_exp, oracle),
                "random_expected": rand_exp, "oracle": oracle, **first[i],
            })
    return pd.DataFrame(rows)


CONDITIONS = [(0, True), (0, False), (5, True), (5, False)]  # (warmup, grupo visible)


def evaluate_conditions(policy: BatchPolicy, users: list[User], n_items: int, name: str,
                        conditions=CONDITIONS, seeds=(0, 1, 2), horizon: int = 20) -> pd.DataFrame:
    frames = []
    for warmup, visible in conditions:
        cfg = SessionConfig(horizon=horizon, warmup=warmup, group_visible=visible)
        df = evaluate_batched(policy, users, n_items, cfg, seeds)
        frames.append(df.assign(policy=name, warmup=warmup, group_visible=visible))
    return pd.concat(frames, ignore_index=True)


def mean_normalized(df: pd.DataFrame) -> float:
    """Normalizado medio por usuario (promediando antes las semillas de evaluación)."""
    keys = [c for c in ["policy", "warmup", "group_visible"] if c in df.columns]
    per_user = df.groupby(["user_id", *keys])["normalized"].mean()
    return float(per_user.groupby(level=keys).mean().mean()) if keys else float(per_user.mean())


def per_step_rewards(policy: BatchPolicy, users: list[User], n_items: int, config: SessionConfig,
                     seed: int = 0) -> np.ndarray:
    """Rating obtenido en cada paso controlado, (n_users, pasos). Para diagnóstico."""
    rngs = [np.random.default_rng([seed, i]) for i in range(len(users))]
    sessions = [Session(u, n_items, config, r) for u, r in zip(users, rngs)]
    policy.reset_batch([s.visible_group for s in sessions], config.horizon)
    out = []
    while not sessions[0].done:
        obs = [s.observed() for s in sessions]
        sc = np.asarray(policy.scores_batch(np.stack([o[0] for o in obs]), np.stack([o[1] for o in obs])))
        out.append([s.step(choose(sc[i], s.available_mask(), rngs[i])) for i, s in enumerate(sessions)])
    return np.array(out).T


class BaselineBatch:
    """Adapta una línea base del protocolo (una sesión a la vez) a la interfaz por lotes."""

    def __init__(self, make):
        self.make = make

    def reset_batch(self, groups, horizon):
        self.ps = [self.make() for _ in groups]
        for p, g in zip(self.ps, groups):
            p.reset(g, horizon)

    def scores_batch(self, items_seen, ratings_seen):
        return np.stack([p.scores(i, r) for p, i, r in zip(self.ps, items_seen, ratings_seen)])
