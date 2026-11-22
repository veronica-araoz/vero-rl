"""Tests de los modelos propios: no fuga en el DT, máscaras, IQL y evaluación por lotes."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocolo.datos import User  # noqa: E402
from protocolo.evaluar import evaluate  # noqa: E402
from protocolo.simulador import SessionConfig  # noqa: E402
from src.dt import DecisionTransformer, DTPolicy, prev_ratings_index, seen_before_mask  # noqa: E402
from src.iql import IQL, IQLPolicy, expectile_loss, iql_losses  # noqa: E402
from src.utils import SinglePolicyAdapter, evaluate_batched  # noqa: E402

N_ITEMS, T = 30, 20


def _batch(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = torch.stack([torch.randperm(N_ITEMS, generator=g)[:T] for _ in range(B)])
    ratings = torch.randint(1, 6, (B, T), generator=g).float()
    rtg = ratings.flip(1).cumsum(1).flip(1)
    groups = torch.randint(0, 9, (B,), generator=g)
    return {"items": items, "ratings": ratings, "rtg": rtg, "groups": groups}


def test_dt_prediccion_en_t_ignora_tokens_actuales_y_futuros():
    torch.manual_seed(0)
    model = DecisionTransformer(N_ITEMS, T, d_model=32, n_layers=2, n_heads=2).eval()
    b = _batch()
    for t in (0, 7, 19):
        base = model(b["rtg"], b["items"], prev_ratings_index(b["ratings"]), b["groups"])[:, t]
        items2 = b["items"].clone()
        items2[:, t:] = torch.randint(0, N_ITEMS, items2[:, t:].shape)
        rtg2 = b["rtg"].clone()
        rtg2[:, t + 1:] = 0.0
        ratings2 = b["ratings"].clone()
        ratings2[:, t:] = torch.randint(1, 6, ratings2[:, t:].shape).float()   # cambia s_{>t}
        out = model(rtg2, items2, prev_ratings_index(ratings2), b["groups"])[:, t]
        assert torch.allclose(base, out, atol=1e-5)


def test_dt_estado_no_incluye_rating_actual():
    r = torch.tensor([[4.0, 2.0, 5.0]])
    assert prev_ratings_index(r).tolist() == [[0, 4, 2]]


def test_seen_before_mask():
    m = seen_before_mask(torch.tensor([[17, 3, 9]]), 20)
    assert not m[0, 0].any()
    assert m[0, 1].nonzero().flatten().tolist() == [17]
    assert m[0, 2].nonzero().flatten().tolist() == [3, 17]


def test_expectil():
    q = torch.tensor([1.0, 2.0, 3.0, 10.0])
    for tau, esperado in [(0.5, 4.0), (0.9, 8.0)]:
        v = torch.zeros(1, requires_grad=True)
        opt = torch.optim.SGD([v], lr=0.05)
        for _ in range(3000):
            opt.zero_grad()
            expectile_loss(q - v, tau).backward()
            opt.step()
        assert abs(v.item() - esperado) < 0.05


def test_iql_gradientes_separados_y_terminal():
    torch.manual_seed(0)
    model = IQL(N_ITEMS, T, d=16)
    b = _batch()
    losses, _ = iql_losses(model, b, gamma=0.9, tau=0.7, beta=3.0)
    for name, net in [("v", model.v), ("q", model.q), ("pi", model.pi)]:
        model.zero_grad()
        losses[name].backward(retain_graph=True)
        others = [n for n in ("v", "q", "pi") if n != name]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
        for o in others:
            assert all(p.grad is None or p.grad.abs().sum() == 0 for p in getattr(model, o).parameters())
    # con γ = 0 el objetivo de Q es sólo la recompensa
    model.zero_grad()
    q_pred = model.q(b["items"], b["ratings"], b["groups"])[:, :T].gather(-1, b["items"].unsqueeze(-1)).squeeze(-1)
    l0, _ = iql_losses(model, b, gamma=0.0, tau=0.7, beta=3.0)
    expected = torch.nn.functional.mse_loss(q_pred, (b["ratings"] - 3) / 2)
    assert torch.allclose(l0["q"], expected, atol=1e-6)


def _users(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return [User(i, int(rng.integers(8)), np.arange(N_ITEMS), rng.integers(1, 6, N_ITEMS).astype(float))
            for i in range(n)]


def test_evaluacion_por_lotes_igual_a_la_oficial():
    torch.manual_seed(0)
    users = _users()
    policies = [DTPolicy(DecisionTransformer(N_ITEMS, T, 32, 2, 2), 80.0),
                IQLPolicy(IQL(N_ITEMS, T, 16), "pi"), IQLPolicy(IQL(N_ITEMS, T, 16), "q")]
    for policy in policies:
        for warmup, visible in [(0, True), (0, False), (5, True)]:
            cfg = SessionConfig(20, warmup, visible)
            a = evaluate(SinglePolicyAdapter(policy), users, N_ITEMS, cfg, seeds=(0, 1))
            b = evaluate_batched(policy, users, N_ITEMS, cfg, seeds=(0, 1))
            cols = ["user_id", "seed", "return", "normalized", "hit_rate", "ndcg@10"]
            assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), atol=1e-6, equal_nan=True)
