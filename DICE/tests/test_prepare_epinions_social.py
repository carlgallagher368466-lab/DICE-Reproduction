from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prepare_epinions_social import (
    convert_raw_epinions,
    normalize_epinions_ratings,
    normalize_epinions_trust,
)


def test_normalize_epinions_ratings_uses_user_item_rating_and_timestamp():
    raw = pd.DataFrame(
        [
            [10, 100, 3, 5, 2, 973234800],
            [11, 101, 7, 4, 1, 973235000],
        ]
    )

    ratings = normalize_epinions_ratings(raw)

    assert ratings.to_dict("records") == [
        {"user": 10, "item": 100, "rating": 5, "ts": 973234800},
        {"user": 11, "item": 101, "rating": 4, "ts": 973235000},
    ]


def test_normalize_epinions_trust_uses_source_and_target_user():
    raw = pd.DataFrame([[10, 20], [20, 30], [30, 30]])

    trust = normalize_epinions_trust(raw)

    assert trust.to_dict("records") == [
        {"user": 10, "friend": 20},
        {"user": 20, "friend": 30},
        {"user": 30, "friend": 30},
    ]


def test_convert_raw_epinions_writes_dice_and_social_files(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "epinions" / "output"
    raw.mkdir()
    (raw / "ratings.txt").write_text(
        "\n".join(
            [
                "10 100 1 5 1 1",
                "10 101 1 5 1 2",
                "10 102 1 4 1 3",
                "20 100 1 5 1 1",
                "20 101 1 4 1 2",
                "20 103 1 5 1 3",
                "30 100 1 5 1 1",
                "30 102 1 5 1 2",
                "30 103 1 4 1 3",
            ]
        ),
        encoding="utf-8",
    )
    (raw / "trust.txt").write_text("10 20\n20 30\n30 40\n", encoding="utf-8")

    summary = convert_raw_epinions(
        rating_path=raw / "ratings.txt",
        trust_path=raw / "trust.txt",
        output_dir=out,
        min_rating=4,
        min_user_interactions=1,
        min_item_interactions=1,
        test_frac=1 / 3,
        val_frac=1 / 3,
        skew_frac=1 / 3,
        seed=3,
    )

    expected_files = {
        "record.csv",
        "train_record.csv",
        "train_skew_record.csv",
        "val_record.csv",
        "test_record.csv",
        "train_coo_record.npz",
        "train_skew_coo_record.npz",
        "train_blend_coo_adj_graph.npz",
        "popularity.npy",
        "social_edges.csv",
        "social_adj.npz",
        "epinions_summary.json",
    }
    assert expected_files.issubset({p.name for p in out.iterdir()})
    assert summary["dataset"] == "epinions"
    assert summary["n_user"] == 3
    assert summary["n_social_edges"] == 2
    assert sp.load_npz(out / "social_adj.npz").shape == (3, 3)
    assert np.load(out / "popularity.npy").shape == (4,)
