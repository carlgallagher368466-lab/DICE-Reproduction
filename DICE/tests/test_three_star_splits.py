from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from make_three_star_splits import (
    build_bipartite_adj,
    build_coo,
    compute_popularity,
    popularity_weighted_user_sample,
    write_variant,
)


def toy_records():
    return pd.DataFrame(
        [
            {"uid": 0, "iid": 0, "ts": 10},
            {"uid": 0, "iid": 1, "ts": 11},
            {"uid": 0, "iid": 2, "ts": 12},
            {"uid": 1, "iid": 0, "ts": 20},
            {"uid": 1, "iid": 1, "ts": 21},
            {"uid": 1, "iid": 3, "ts": 22},
            {"uid": 2, "iid": 0, "ts": 30},
            {"uid": 2, "iid": 2, "ts": 31},
            {"uid": 2, "iid": 3, "ts": 32},
        ]
    )


def test_popularity_weighted_user_sample_keeps_users_and_ratio():
    record = toy_records()
    popularity = compute_popularity(record, n_item=4)

    sample = popularity_weighted_user_sample(
        record,
        popularity=popularity,
        frac=1 / 3,
        mode="inverse",
        cap_percentile=None,
        seed=7,
    )

    assert set(sample["uid"]) == {0, 1, 2}
    assert sample.groupby("uid").size().to_dict() == {0: 1, 1: 1, 2: 1}
    assert set(sample.columns) == {"uid", "iid", "ts"}


def test_build_coo_preserves_shape_and_interactions():
    record = toy_records().iloc[:4]
    coo = build_coo(record, n_user=3, n_item=4)

    assert coo.shape == (3, 4)
    assert coo.nnz == 4
    assert coo.tocsr()[0, 2] == 1


def test_build_bipartite_adj_uses_user_item_offsets():
    coo = build_coo(toy_records().iloc[:2], n_user=3, n_item=4)
    adj = build_bipartite_adj(coo)

    assert adj.shape == (7, 7)
    assert adj.tocsr()[0, 3] == 1
    assert adj.tocsr()[3, 0] == 1
    assert adj.nnz == 4


def test_write_variant_outputs_dice_compatible_files(tmp_path):
    record = toy_records()
    train = record.iloc[:4].copy()
    train_skew = record.iloc[4:6].copy()
    val = record.iloc[6:7].copy()
    test = record.iloc[7:].copy()

    write_variant(
        output_dir=tmp_path,
        full_record=record,
        train_record=train,
        train_skew_record=train_skew,
        val_record=val,
        test_record=test,
        n_user=3,
        n_item=4,
    )

    expected = {
        "record.csv",
        "train_record.csv",
        "train_skew_record.csv",
        "val_record.csv",
        "test_record.csv",
        "coo_record.npz",
        "train_coo_record.npz",
        "train_skew_coo_record.npz",
        "val_coo_record.npz",
        "test_coo_record.npz",
        "train_coo_adj_graph.npz",
        "train_skew_coo_adj_graph.npz",
        "train_blend_coo_adj_graph.npz",
        "popularity.npy",
        "popularity_all.npy",
        "popularity_skew.npy",
        "popularity_blend.npy",
    }
    assert expected.issubset({p.name for p in tmp_path.iterdir()})

    assert sp.load_npz(tmp_path / "train_coo_record.npz").shape == (3, 4)
    assert np.load(tmp_path / "popularity.npy").shape == (4,)
