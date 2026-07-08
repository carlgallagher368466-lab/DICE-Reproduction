import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import model


def test_idice_forward_returns_scalar_loss_with_social_edges():
    net = model.IDICE(
        num_users=5,
        num_items=7,
        embedding_size=8,
        dis_loss="L2",
        dis_pen=0.01,
        int_weight=0.1,
        pop_weight=0.1,
        social_weight=0.2,
        social_reg_weight=0.05,
    )
    users = torch.tensor([[0, 1], [2, 3]])
    pos = torch.tensor([[1, 2], [3, 4]])
    neg = torch.tensor([[4, 5], [0, 6]])
    mask = torch.tensor([[True, False], [False, True]])
    social_edges = torch.tensor([[0, 1], [1, 2], [3, 4]], dtype=torch.long)

    loss = net(users, pos, neg, mask, social_edges)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_idice_embeddings_concatenate_three_branches():
    net = model.IDICE(
        num_users=3,
        num_items=4,
        embedding_size=6,
        dis_loss="L2",
        dis_pen=0.01,
        int_weight=0.1,
        pop_weight=0.1,
        social_weight=0.2,
        social_reg_weight=0.05,
    )

    item_embeddings = net.get_item_embeddings()
    user_embeddings = net.get_user_embeddings()

    assert item_embeddings.shape == (4, 18)
    assert user_embeddings.shape == (3, 18)


def test_idice_zero_social_weight_removes_social_embeddings_from_retrieval():
    net = model.IDICE(
        num_users=3,
        num_items=4,
        embedding_size=6,
        dis_loss="L2",
        dis_pen=0.01,
        int_weight=0.1,
        pop_weight=0.1,
        social_weight=0.0,
        social_reg_weight=0.0,
    )

    item_embeddings = net.get_item_embeddings()
    user_embeddings = net.get_user_embeddings()

    assert (item_embeddings[:, 12:] == 0).all()
    assert (user_embeddings[:, 12:] == 0).all()
