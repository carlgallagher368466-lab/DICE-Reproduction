"""Prepare Epinions social recommendation data in DICE-compatible format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import scipy.sparse as sp

from prepare_ciao_social import (
    build_social_adj,
    filter_interactions,
    reindex_ciao,
    _split_records,
)
from make_three_star_splits import write_variant


def read_raw_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, sep=r"\s+", engine="python", header=None, comment="#")


def normalize_epinions_ratings(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize MSU Epinions timestamp file.

    The source file has six numeric columns. For recommendation experiments,
    DICE needs user id, item id, binary-positive rating signal, and timestamp.
    The fourth source column is the 1-5 score and the sixth is Unix time.
    """

    if raw.shape[1] < 6:
        raise ValueError("Epinions rating file must have at least 6 columns")
    ratings = raw.iloc[:, [0, 1, 3, 5]].copy()
    ratings.columns = ["user", "item", "rating", "ts"]
    return ratings.apply(pd.to_numeric, errors="raise").astype(
        {"user": "int64", "item": "int64", "rating": "int64", "ts": "int64"}
    )


def normalize_epinions_trust(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.shape[1] < 2:
        raise ValueError("Epinions trust file must have at least 2 columns")
    trust = raw.iloc[:, [0, 1]].copy()
    trust.columns = ["user", "friend"]
    return trust.apply(pd.to_numeric, errors="raise").astype({"user": "int64", "friend": "int64"})


def read_epinions_ratings(path: Path) -> pd.DataFrame:
    return normalize_epinions_ratings(read_raw_table(path))


def read_epinions_trust(path: Path) -> pd.DataFrame:
    return normalize_epinions_trust(read_raw_table(path))


def convert_raw_epinions(
    rating_path: Path,
    trust_path: Path,
    output_dir: Path,
    min_rating: float,
    min_user_interactions: int,
    min_item_interactions: int,
    test_frac: float,
    val_frac: float,
    skew_frac: float,
    seed: int,
) -> dict[str, int | float | str]:
    ratings = read_epinions_ratings(rating_path)
    trust = read_epinions_trust(trust_path)
    ratings = filter_interactions(ratings, min_rating, min_user_interactions, min_item_interactions)
    record, social_edges, user_map, item_map = reindex_ciao(ratings, trust)

    n_user = len(user_map)
    n_item = len(item_map)
    train_record, train_skew_record, val_record, test_record = _split_records(
        record, n_item=n_item, test_frac=test_frac, val_frac=val_frac, skew_frac=skew_frac, seed=seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_variant(output_dir, record, train_record, train_skew_record, val_record, test_record, n_user, n_item)
    social_edges.to_csv(output_dir / "social_edges.csv", index=False)
    sp.save_npz(output_dir / "social_adj.npz", build_social_adj(social_edges, n_user))
    (output_dir / "user_reindex.json").write_text(json.dumps(user_map, indent=2), encoding="utf-8")
    (output_dir / "item_reindex.json").write_text(json.dumps(item_map, indent=2), encoding="utf-8")

    summary = {
        "dataset": "epinions",
        "n_user": int(n_user),
        "n_item": int(n_item),
        "n_interactions": int(len(record)),
        "n_social_edges": int(len(social_edges)),
        "n_train": int(len(train_record)),
        "n_train_skew": int(len(train_skew_record)),
        "n_val": int(len(val_record)),
        "n_test": int(len(test_record)),
        "min_rating": float(min_rating),
    }
    (output_dir / "epinions_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rating-path", type=Path, required=True)
    parser.add_argument("--trust-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/epinions/output"))
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--skew-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    summary = convert_raw_epinions(
        rating_path=args.rating_path,
        trust_path=args.trust_path,
        output_dir=args.output_dir,
        min_rating=args.min_rating,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        skew_frac=args.skew_frac,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
