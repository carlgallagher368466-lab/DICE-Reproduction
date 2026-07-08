import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import model


def make_graph():
    edges = torch.tensor(
        [
            [0, 3],
            [3, 0],
            [1, 4],
            [4, 1],
            [2, 5],
            [5, 2],
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
        ],
        dtype=torch.long,
    )
    values = torch.ones(edges.size(0))
    adj = torch.sparse_coo_tensor(edges.t(), values, (6, 6)).coalesce()
    degree = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)
    norm_values = torch.pow(degree[edges[:, 0]], -0.5) * torch.pow(degree[edges[:, 1]], -0.5)
    return torch.sparse_coo_tensor(edges.t(), norm_values, (6, 6)).coalesce()


def test_ngcf_conv_has_feature_transforms_and_preserves_shape():
    conv = model.NGCFConv(embedding_size=4, dropout=0.0)
    graph = make_graph()
    features = torch.randn(6, 4)

    out = conv(graph, features, training=True)

    assert out.shape == (6, 4)
    assert torch.isfinite(out).all()
    assert hasattr(conv, "linear_gc")
    assert hasattr(conv, "linear_bi")


def test_ngcf_pair_forward_shapes():
    net = model.NGCF(num_users=3, num_items=3, embedding_size=4, num_layers=2, dropout=0.0)
    graph = make_graph()
    users = torch.tensor([[0, 1], [2, 0]])
    pos = torch.tensor([[0, 1], [2, 1]])
    neg = torch.tensor([[2, 0], [1, 2]])

    p_score, n_score = net.pair_forward(users, pos, neg, graph)

    assert p_score.shape == (2, 2)
    assert n_score.shape == (2, 2)
    assert torch.isfinite(p_score).all()
    assert torch.isfinite(n_score).all()


def test_ngcfdice_forward_returns_scalar_loss():
    net = model.NGCFDICE(
        num_users=3,
        num_items=3,
        embedding_size=4,
        num_layers=2,
        dropout=0.0,
        dis_loss="L2",
        dis_pen=0.01,
        int_weight=0.1,
        pop_weight=0.1,
    )
    graph = make_graph()
    users = torch.tensor([[0, 1], [2, 0]])
    pos = torch.tensor([[0, 1], [2, 1]])
    neg = torch.tensor([[2, 0], [1, 2]])
    mask = torch.tensor([[True, False], [False, True]])

    loss = net(users, pos, neg, mask, graph)

    assert loss.shape == ()
    assert torch.isfinite(loss)
