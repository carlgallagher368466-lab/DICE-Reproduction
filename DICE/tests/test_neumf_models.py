import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import model


def test_neumf_pair_forward_shapes():
    net = model.NeuMF(
        num_users=5,
        num_items=7,
        embedding_size=8,
        mlp_layers=[16, 8],
        dropout=0.0,
    )
    users = torch.tensor([[0, 1], [2, 3]])
    pos = torch.tensor([[1, 2], [3, 4]])
    neg = torch.tensor([[4, 5], [0, 6]])

    p_score, n_score = net.pair_forward(users, pos, neg)

    assert p_score.shape == (2, 2)
    assert n_score.shape == (2, 2)
    assert torch.isfinite(p_score).all()
    assert torch.isfinite(n_score).all()


def test_neumfdice_forward_returns_scalar_loss():
    net = model.NeuMFDICE(
        num_users=5,
        num_items=7,
        embedding_size=8,
        mlp_layers=[16, 8],
        dropout=0.0,
        dis_loss="L2",
        dis_pen=0.01,
        int_weight=0.1,
        pop_weight=0.1,
    )
    users = torch.tensor([[0, 1], [2, 3]])
    pos = torch.tensor([[1, 2], [3, 4]])
    neg = torch.tensor([[4, 5], [0, 6]])
    mask = torch.tensor([[True, False], [False, True]])

    loss = net(users, pos, neg, mask)

    assert loss.shape == ()
    assert torch.isfinite(loss)
