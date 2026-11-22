"""Experimentos del TP. Se ejecuta desde la raíz del repo:

    python -m src.experimentos dt_train   --logs random --seeds 0 1 2
    python -m src.experimentos dt_select  --logs random
    python -m src.experimentos iql_sweep  --logs random
    python -m src.experimentos iql_train  --logs random --gamma 0.9 --tau 0.7 --beta 3 --seeds 0 1 2
    python -m src.experimentos test_eval  --logs random
    python -m src.experimentos rc_test    --logs random

Todo se guarda en outputs/. La selección (retorno pedido del DT; τ, β y γ de
IQL) se hace sólo con validación. Test se usa en `test_eval` y `rc_test`.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from protocolo.datos import load_test, load_train, num_items, split_by_user
from protocolo.evaluar import write_reports
from protocolo.logs import logged_episodes, logs_cache

from .dt import DecisionTransformer, DTConfig, DTPolicy, train_dt
from .iql import IQL, IQLConfig, IQLPolicy, train_iql
from .utils import OUTPUTS, episodes_to_tensors, evaluate_conditions, mean_normalized, save_json

DATASET = "netflix"
HORIZON = 20
VAL_CONDITIONS = [(0, True), (0, False)]
DT_DEFAULT = dict(d_model=64, n_layers=2, n_heads=4, epochs=6)
IQL_DEFAULT = dict(d=64, epochs=5)
IQL_GRID = dict(gamma=(0.0, 0.9), tau=(0.7, 0.9), beta=(3.0, 10.0))

torch.set_num_threads(max(1, torch.get_num_threads()))


def load(logs: str):
    n = num_items(DATASET)
    train, val = split_by_user(load_train(DATASET))
    tr = logged_episodes(logs, train, train, n, HORIZON, seed=0, cache=logs_cache(DATASET, logs, "train"))
    va = logged_episodes(logs, val, train, n, HORIZON, seed=1, cache=logs_cache(DATASET, logs, "val"))
    return n, train, val, tr, va


def dt_dir(logs, seed):
    return OUTPUTS / "dt" / logs / f"seed{seed}"


def iql_dir(logs, cfg: IQLConfig):
    return OUTPUTS / "iql" / logs / f"g{cfg.gamma}_t{cfg.tau}_b{cfg.beta}" / f"seed{cfg.seed}"


def load_dt(logs, seed, n):
    d = dt_dir(logs, seed)
    cfg = DTConfig(**json.loads((d / "config.json").read_text())["config"])
    model = DecisionTransformer(n, HORIZON, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.dropout)
    model.load_state_dict(torch.load(d / "model.pt"))
    return model.eval()


def load_iql(path, n):
    cfg = IQLConfig(**json.loads((path / "config.json").read_text())["config"])
    model = IQL(n, HORIZON, cfg.d)
    model.load_state_dict(torch.load(path / "model.pt"))
    return model.eval(), cfg


# ----------------------------------------------------------------------------- DT

def dt_train(logs, seeds):
    n, _, _, tr, va = load(logs)
    train_data, val_data = episodes_to_tensors(tr.episodes), episodes_to_tensors(va.episodes)
    for seed in seeds:
        cfg = DTConfig(**DT_DEFAULT, seed=seed)
        model, hist = train_dt(train_data, val_data, n, cfg)
        d = dt_dir(logs, seed)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / "model.pt")
        save_json({"config": asdict(cfg), "logs": logs, "n_train_episodes": len(tr.episodes.groups),
                   "uniform_ce": float(np.log(n))}, d / "config.json")
        save_json(hist, d / "history.json")


def train_return_support(logs):
    _, _, _, tr, _ = load(logs)
    R = tr.episodes.ratings.sum(1)
    return R


def dt_candidates(R):
    pct = [10, 25, 50, 75, 90, 99]
    vals = {f"p{p}": float(np.percentile(R, p)) for p in pct}
    vals["max"] = float(R.max())
    vals["max+10"] = float(R.max() + 10)
    return vals


def dt_select(logs, seeds=(0, 1, 2)):
    n, _, val, tr, _ = load(logs)
    cands = dt_candidates(tr.episodes.ratings.sum(1))
    rows = []
    for seed in seeds:
        model = load_dt(logs, seed, n)
        for name, target in cands.items():
            df = evaluate_conditions(DTPolicy(model, target), val, n, f"dt_{name}", VAL_CONDITIONS, seeds=(0,))
            for (w, vis), g in df.groupby(["warmup", "group_visible"]):
                rows.append({"train_seed": seed, "candidate": name, "target": target, "warmup": w,
                             "group_visible": vis, "normalized": mean_normalized(g),
                             "return": g["return"].mean()})
            print(f"[val DT {logs}] seed {seed} {name}={target:.0f} → {rows[-1]['normalized']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    out = OUTPUTS / "seleccion" / f"dt_{logs}_val.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    agg = res.groupby(["candidate", "target"])["normalized"].mean().sort_values(ascending=False)
    best = agg.index[0]
    save_json({"candidate": best[0], "target": best[1], "val_normalized": float(agg.iloc[0]),
               "criterio": "media de normalizado en val (warm-up 0, grupo visible y oculto, 3 semillas)"},
              OUTPUTS / "seleccion" / f"dt_{logs}_elegido.json")
    print(agg)


# ----------------------------------------------------------------------------- IQL

def _train_iql_one(logs, cfg, train_data):
    n = num_items(DATASET)
    model, hist = train_iql(train_data, n, cfg)
    d = iql_dir(logs, cfg)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    save_json({"config": asdict(cfg), "logs": logs}, d / "config.json")
    save_json(hist, d / "history.json")
    return model


def iql_sweep(logs):
    n, _, val, tr, _ = load(logs)
    train_data = episodes_to_tensors(tr.episodes)
    rows = []
    for gamma, tau, beta in itertools.product(*IQL_GRID.values()):
        cfg = IQLConfig(**IQL_DEFAULT, gamma=gamma, tau=tau, beta=beta, seed=0)
        path = iql_dir(logs, cfg)
        model = load_iql(path, n)[0] if (path / "model.pt").exists() else _train_iql_one(logs, cfg, train_data)
        for mode in ("pi", "q"):
            df = evaluate_conditions(IQLPolicy(model, mode), val, n, mode, VAL_CONDITIONS, seeds=(0,))
            for (w, vis), g in df.groupby(["warmup", "group_visible"]):
                rows.append({"gamma": gamma, "tau": tau, "beta": beta, "mode": mode, "warmup": w,
                             "group_visible": vis, "normalized": mean_normalized(g)})
        print(f"[val IQL] γ={gamma} τ={tau} β={beta}: π {np.mean([r['normalized'] for r in rows[-4:-2]]):.3f} "
              f"maxQ {np.mean([r['normalized'] for r in rows[-2:]]):.3f}", flush=True)
    res = pd.DataFrame(rows)
    (OUTPUTS / "seleccion").mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTPUTS / "seleccion" / f"iql_{logs}_val.csv", index=False)
    agg = res[res["mode"] == "pi"].groupby(["gamma", "tau", "beta"])["normalized"].mean().sort_values(ascending=False)
    g, t, b = agg.index[0]
    best_by_gamma = {str(gm): list(map(float, agg.xs(gm, level="gamma").index[0])) for gm in IQL_GRID["gamma"]}
    save_json({"gamma": g, "tau": t, "beta": b, "val_normalized_pi": float(agg.iloc[0]),
               "mejor_tau_beta_por_gamma": best_by_gamma,
               "criterio": "media de normalizado de π en val (warm-up 0, grupo visible y oculto, semilla 0)"},
              OUTPUTS / "seleccion" / f"iql_{logs}_elegido.json")
    print(agg)


def iql_train(logs, gamma, tau, beta, seeds):
    _, _, _, tr, _ = load(logs)
    train_data = episodes_to_tensors(tr.episodes)
    for seed in seeds:
        cfg = IQLConfig(**IQL_DEFAULT, gamma=gamma, tau=tau, beta=beta, seed=seed)
        if (iql_dir(logs, cfg) / "model.pt").exists():
            print("ya existe", iql_dir(logs, cfg))
            continue
        _train_iql_one(logs, cfg, train_data)


# ----------------------------------------------------------------------------- test

def test_eval(logs, seeds=(0, 1, 2)):
    """Evaluación final en test: DT (retorno elegido en val), IQL π y máximo de Q, por semilla."""
    n = num_items(DATASET)
    test = load_test(DATASET)
    sel_dt = json.loads((OUTPUTS / "seleccion" / f"dt_{logs}_elegido.json").read_text())
    sel_iql = json.loads((OUTPUTS / "seleccion" / "iql_random_elegido.json").read_text())
    gammas = [0.0, 0.9] if logs == "random" else [sel_iql["gamma"]]
    for seed in seeds:
        out = OUTPUTS / "test" / logs / f"seed{seed}"
        frames = [evaluate_conditions(DTPolicy(load_dt(logs, seed, n), sel_dt["target"]), test, n, "dt")]
        for gm in gammas:
            t, b = sel_iql["mejor_tau_beta_por_gamma"][str(gm)] if gm != sel_iql["gamma"] else \
                (sel_iql["tau"], sel_iql["beta"])
            cfg = IQLConfig(**IQL_DEFAULT, gamma=gm, tau=t, beta=b, seed=seed)
            model, _ = load_iql(iql_dir(logs, cfg), n)
            suffix = "" if gm == sel_iql["gamma"] else f"_gamma{gm}"
            frames.append(evaluate_conditions(IQLPolicy(model, "pi"), test, n, f"iql_pi{suffix}"))
            frames.append(evaluate_conditions(IQLPolicy(model, "q"), test, n, f"iql_maxq{suffix}"))
        write_reports(pd.concat(frames, ignore_index=True), out)


def rc_test(logs, seeds=(0, 1, 2)):
    """Return conditioning en test: retornos pedidos vs obtenidos, por grupo."""
    n, _, _, tr, _ = load(logs)
    test = load_test(DATASET)
    cands = dt_candidates(tr.episodes.ratings.sum(1))
    extra = {f"R{v}": float(v) for v in (40, 50, 60, 70, 80, 90, 100)}
    cands = {**extra, **cands}
    frames = []
    for seed in seeds:
        model = load_dt(logs, seed, n)
        for name, target in cands.items():
            df = evaluate_conditions(DTPolicy(model, target), test, n, "dt", VAL_CONDITIONS, seeds=(0,))
            frames.append(df.assign(train_seed=seed, candidate=name, target=target))
        print(f"[RC {logs}] seed {seed} listo", flush=True)
    res = pd.concat(frames, ignore_index=True)
    out = OUTPUTS / "return_conditioning" / logs
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "sesiones.csv.gz", index=False)
    save_json({"train_returns": {"min": float(tr.episodes.ratings.sum(1).min()),
                                 "max": float(tr.episodes.ratings.sum(1).max()),
                                 "mean": float(tr.episodes.ratings.sum(1).mean())}}, out / "soporte.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("etapa", choices=["dt_train", "dt_select", "iql_sweep", "iql_train", "test_eval", "rc_test"])
    p.add_argument("--logs", default="random", choices=["random", "mixto"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--gamma", type=float)
    p.add_argument("--tau", type=float)
    p.add_argument("--beta", type=float)
    a = p.parse_args()
    if a.etapa == "dt_train":
        dt_train(a.logs, a.seeds)
    elif a.etapa == "dt_select":
        dt_select(a.logs, a.seeds)
    elif a.etapa == "iql_sweep":
        iql_sweep(a.logs)
    elif a.etapa == "iql_train":
        iql_train(a.logs, a.gamma, a.tau, a.beta, a.seeds)
    elif a.etapa == "test_eval":
        test_eval(a.logs, a.seeds)
    elif a.etapa == "rc_test":
        rc_test(a.logs, a.seeds)


if __name__ == "__main__":
    main()
