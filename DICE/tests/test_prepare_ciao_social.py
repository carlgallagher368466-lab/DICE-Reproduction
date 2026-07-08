from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prepare_ciao_social import (
    build_social_adj,
    convert_raw_ciao,
    filter_interactions,
    normalize_librec_ratings,
    normalize_librec_trust,
    parse_table,
    reindex_ciao,
)


def test_parse_table_accepts_whitespace_and_comma_files(tmp_path):
    whitespace = tmp_path / "ratings.txt"
    whitespace.write_text("10 100 5 111\n11 101 4 112\n", encoding="utf-8")
    comma = tmp_path / "trust.csv"
    comma.write_text("10,11\n11,12\n", encoding="utf-8")

    ratings = parse_table(whitespace, ["user", "item", "rating", "ts"])
    trust = parse_table(comma, ["user", "friend"])

    assert ratings.to_dict("records")[0] == {"user": 10, "item": 100, "rating": 5, "ts": 111}
    assert trust.to_dict("records") == [{"user": 10, "friend": 11}, {"user": 11, "friend": 12}]


def test_filter_interactions_keeps_positive_and_min_counts():
    ratings = pd.DataFrame(
        [
            {"user": 1, "item": 10, "rating": 5, "ts": 1},
            {"user": 1, "item": 11, "rating": 4, "ts": 2},
            {"user": 2, "item": 10, "rating": 5, "ts": 3},
            {"user": 2, "item": 11, "rating": 4, "ts": 4},
            {"user": 3, "item": 12, "rating": 5, "ts": 5},
        ]
    )

    filtered = filter_interactions(ratings, min_rating=4, min_user_interactions=2, min_item_interactions=2)

    assert filtered[["user", "item"]].to_dict("records") == [
        {"user": 1, "item": 10},
        {"user": 1, "item": 11},
        {"user": 2, "item": 10},
        {"user": 2, "item": 11},
    ]


def test_normalize_librec_ratings_uses_user_item_rating_and_review_date():
    raw = pd.DataFrame(
        [
            [10, 100, 7, 900, 5, "2001-01-01"],
            [11, 101, 8, 901, 3, "2001-01-02"],
        ]
    )

    ratings = normalize_librec_ratings(raw)

    assert ratings[["user", "item", "rating"]].to_dict("records") == [
        {"user": 10, "item": 100, "rating": 5},
        {"user": 11, "item": 101, "rating": 3},
    ]
    assert ratings["ts"].iloc[1] > ratings["ts"].iloc[0]


def test_normalize_librec_trust_uses_trustor_and_trustee():
    raw = pd.DataFrame([[10, 11, 1], [12, 13, 0]])

    trust = normalize_librec_trust(raw)

    assert trust.to_dict("records") == [{"user": 10, "friend": 11}, {"user": 12, "friend": 13}]


def test_reindex_ciao_aligns_trust_to_rating_users():
    ratings = pd.DataFrame(
        [
            {"user": 10, "item": 100, "rating": 5, "ts": 1},
            {"user": 20, "item": 100, "rating": 5, "ts": 2},
            {"user": 20, "item": 200, "rating": 5, "ts": 3},
        ]
    )
    trust = pd.DataFrame([{"user": 10, "friend": 20}, {"user": 20, "friend": 30}])

    record, social_edges, user_map, item_map = reindex_ciao(ratings, trust)

    assert set(record.columns) == {"uid", "iid", "ts"}
    assert set(record["uid"]) == {0, 1}
    assert set(record["iid"]) == {0, 1}
    assert social_edges.to_dict("records") == [{"src": user_map[10], "dst": user_map[20]}]
    assert item_map == {100: 0, 200: 1}


def test_build_social_adj_is_symmetric_and_keeps_shape():
    edges = pd.DataFrame([{"src": 0, "dst": 1}, {"src": 1, "dst": 2}])
    adj = build_social_adj(edges, n_user=4)

    assert adj.shape == (4, 4)
    assert adj.tocsr()[0, 1] == 1
    assert adj.tocsr()[1, 0] == 1
    assert adj.tocsr()[3, 3] == 0


def test_convert_raw_ciao_writes_dice_and_social_files(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "ciao" / "output"
    raw.mkdir()
    (raw / "ratings.txt").write_text(
        "\n".join(
            [
                "10 100 5 1",
                "10 101 5 2",
                "10 102 4 3",
                "20 100 5 1",
                "20 101 4 2",
                "20 103 5 3",
                "30 100 5 1",
                "30 102 5 2",
                "30 103 4 3",
            ]
        ),
        encoding="utf-8",
    )
    (raw / "trust.txt").write_text("10 20\n20 30\n30 40\n", encoding="utf-8")

    summary = convert_raw_ciao(
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
        "ciao_summary.json",
    }
    assert expected_files.issubset({p.name for p in out.iterdir()})
    assert summary["n_user"] == 3
    assert summary["n_social_edges"] == 2
    assert sp.load_npz(out / "social_adj.npz").shape == (3, 3)
    assert np.load(out / "popularity.npy").shape == (4,)
