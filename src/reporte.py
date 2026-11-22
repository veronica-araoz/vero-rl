"""Funciones para armar tablas y gráficos a partir de outputs/ (los usan los notebooks)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .utils import OUTPUTS

COND_LABELS = {(0, True): "visible", (0, False): "oculto", (5, True): "warm-up 5, visible",
               (5, False): "warm-up 5, oculto"}
BASELINES = ["random", "popularity", "item_mean", "group_mean", "group_posterior", "sistema_anterior_eps0"]


def read_sessions(folder) -> pd.DataFrame:
    """Lee sesiones.csv o, si no está, sesiones.csv.gz (versión comprimida que se sube al repo)."""
    folder = OUTPUTS / folder
    path = folder / "sesiones.csv"
    return pd.read_csv(path if path.exists() else folder / "sesiones.csv.gz")


def _per_user(df: pd.DataFrame, metric: str, keys=("policy", "warmup", "group_visible")) -> pd.Series:
    """Promedia primero las semillas de evaluación de cada usuario (como `summarize`)."""
    return df.groupby(["user_id", *keys])[metric].mean()


def by_condition(df: pd.DataFrame, metric: str = "normalized") -> pd.DataFrame:
    """Media e IC 95 % entre usuarios, por política y condición."""
    pu = _per_user(df, metric).reset_index()
    g = pu.groupby(["policy", "warmup", "group_visible"])[metric]
    return pd.DataFrame({"mean": g.mean(), "ci95": 1.96 * g.std(ddof=1) / np.sqrt(g.count())})


def model_seed_table(logs: str, metric: str = "normalized", seeds=(0, 1, 2)) -> pd.DataFrame:
    """Modelos: media entre semillas de entrenamiento, desvío entre semillas e IC 95 % por usuario medio."""
    per_seed, cis = [], []
    for s in seeds:
        df = read_sessions(f"test/{logs}/seed{s}")
        bc = by_condition(df, metric)
        per_seed.append(bc["mean"].rename(s))
        cis.append(bc["ci95"].rename(s))
    m = pd.concat(per_seed, axis=1)
    return pd.DataFrame({"media": m.mean(axis=1), "desvío_semillas": m.std(axis=1, ddof=1),
                         "ic95_usuarios": pd.concat(cis, axis=1).mean(axis=1),
                         **{f"semilla_{s}": m[s] for s in seeds}})


def baseline_table(logs: str, metric: str = "normalized") -> pd.DataFrame:
    df = read_sessions(f"baselines/{logs}_test")
    bc = by_condition(df, metric)
    return pd.DataFrame({"media": bc["mean"], "desvío_semillas": np.nan, "ic95_usuarios": bc["ci95"]})


def main_table(logs: str, metric: str = "normalized", fmt: bool = True) -> pd.DataFrame:
    """Tabla del reporte: filas = políticas, columnas = condiciones."""
    t = pd.concat([baseline_table(logs, metric), model_seed_table(logs, metric)[["media", "desvío_semillas", "ic95_usuarios"]]])
    t = t.reset_index()
    t["condición"] = [COND_LABELS[(w, v)] for w, v in zip(t.warmup, t.group_visible)]
    order = [p for p in BASELINES + ["dt", "iql_pi", "iql_maxq", "iql_pi_gamma0.0", "iql_maxq_gamma0.0",
                                     "iql_pi_gamma0.9", "iql_maxq_gamma0.9"] if p in set(t.policy)]
    if not fmt:
        return t
    def cell(r):
        s = f"{r.media:.3f}"
        if not np.isnan(r.desvío_semillas):
            s += f" ± {r.desvío_semillas:.3f}"
        return s + f" [±{r.ic95_usuarios:.3f}]"
    t["valor"] = t.apply(cell, axis=1)
    piv = t.pivot(index="policy", columns="condición", values="valor")
    return piv.loc[order, list(COND_LABELS.values())]


def group_table(logs: str, policies=None, warmup=0, visible=False, metric="normalized") -> pd.DataFrame:
    """Normalizado medio por grupo (modelos: promedio de las 3 semillas)."""
    frames = [read_sessions(f"baselines/{logs}_test")]
    for s in (0, 1, 2):
        frames.append(read_sessions(f"test/{logs}/seed{s}").assign(train_seed=s))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.warmup == warmup) & (df.group_visible == visible)]
    if policies:
        df = df[df.policy.isin(policies)]
    return df.groupby(["policy", "group"])[metric].mean().unstack("group").round(3)


def load_json(path):
    return json.loads((OUTPUTS / path).read_text(encoding="utf-8"))
