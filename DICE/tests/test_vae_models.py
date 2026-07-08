import os
import sys
import types

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.modules.setdefault("faiss", types.SimpleNamespace())

import model


def test_vae_pair_forward_shapes_and_kl_loss():
    net = model.VAE(num_users=3, num_items=4, embedding_size=6, latent_size=5, hidden_size=8, dropout=0.0)
    users = torch.tensor([[0, 1], [2, 0]])
    pos = torch.tensor([[0, 1], [2, 3]])
    neg = torch.tensor([[3, 2], [1, 0]])

    p_score, n_score, kl_loss = net.pair_forward(users, pos, neg)

    assert p_score.shape == (2, 2)
    assert n_score.shape == (2, 2)
    assert kl_loss.shape == ()
    assert torch.isfinite(p_score).all()
    assert torch.isfinite(n_score).all()
    assert torch.isfinite(kl_loss)
    assert kl_loss >= 0


def test_vaedice_forward_returns_scalar_loss_and_adapts():
    net = model.VAEDICE(
        num_users=3,
        num_items=4,
        embedding_size=6,
        latent_size=5,
        hidden_size=8,
        dropout=0.0,
        dis_loss="L2",
        dis_pen=0.01,
        int_weight=0.1,
        pop_weight=0.1,
        kl_weight=0.001,
    )
    users = torch.tensor([[0, 1], [2, 0]])
    pos = torch.tensor([[0, 1], [2, 3]])
    neg = torch.tensor([[3, 2], [1, 0]])
    mask = torch.tensor([[True, False], [False, True]])

    loss = net(users, pos, neg, mask)
    old_int_weight = net.int_weight
    net.adapt(epoch=1, decay=0.9)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert net.int_weight == old_int_weight * 0.9


def test_vae_score_supports_full_item_scoring():
    net = model.VAE(num_users=3, num_items=4, embedding_size=6, latent_size=5, hidden_size=8, dropout=0.0)
    users = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    items = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])

    scores = net.score(users, items)

    assert scores.shape == (2, 4)
    assert torch.isfinite(scores).all()
