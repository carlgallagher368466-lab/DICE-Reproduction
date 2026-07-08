"""Prepare Ciao social recommendation data in DICE-compatible format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from make_three_star_splits import (
    BASE_COLUMNS,
    build_bipartite_adj,
    build_coo,
    compute_popularity,
    popularity_weighted_user_sample,
    subtract_records,
    temporal_validation_split,
    write_variant,
)


def parse_table(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    frame = pd.read_csv(path, sep=r"[\s,]+", engine="python", header=None, comment="#")
    if frame.shape[1] < len(columns):
        raise ValueError(f"{path} has {frame.shape[1]} columns, expected at least {len(columns)}")
    frame = frame.iloc[:, : len(columns)]
    frame.columns = columns
    return frame.apply(pd.to_numeric, errors="raise")


def normalize_librec_ratings(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.shape[1] < 6:
        raise ValueError("LibRec Ciao rating file must have at least 6 columns")
    ratings = raw.iloc[:, [0, 1, 4, 5]].copy()
    ratings.columns = ["user", "item", "rating", "ts"]
    ratings[["user", "item", "rating"]] = ratings[["user", "item", "rating"]].apply(pd.to_numeric, errors="raise")
    ts_numeric = pd.to_numeric(ratings["ts"], errors="coerce")
    if ts_numeric.isna().any():
        parsed = pd.to_datetime(ratings["ts"], errors="raise")
        ratings["ts"] = (parsed.astype("int64") // 10**9).astype(np.int64)
    else:
        ratings["ts"] = ts_numeric.astype(np.int64)
    return ratings


def normalize_librec_trust(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.shape[1] < 2:
        raise ValueError("LibRec Ciao trust file must have at least 2 columns")
    trust = raw.iloc[:, [0, 1]].copy()
    trust.columns = ["user", "friend"]
    return trust.apply(pd.to_numeric, errors="raise")


def read_ciao_ratings(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"[\s,]+", engine="python", header=None, comment="#")
    if raw.shape[1] >= 6:
        return normalize_librec_ratings(raw)
    if raw.shape[1] >= 4:
        raw = raw.iloc[:, :4].copy()
        raw.columns = ["user", "item", "rating", "ts"]
        return raw.apply(pd.to_numeric, errors="raise")
    raise ValueError(f"{path} has {raw.shape[1]} columns, expected 4 or 6")


def read_ciao_trust(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"[\s,]+", engine="python", header=None, comment="#")
    if raw.shape[1] >= 2:
        return normalize_librec_trust(raw)
    raise ValueError(f"{path} has {raw.shape[1]} columns, expected at least 2")


def filter_interactions(
    ratings: pd.DataFrame,
    min_rating: float,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    required = {"user", "item", "rating", "ts"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"ratings missing required columns: {sorted(missing)}")

    filtered = ratings.loc[ratings["rating"] >= min_rating, ["user", "item", "rating", "ts"]].copy()
    while True:
        before = len(filtered)
        user_counts = filtered.groupby("user").size()
        kept_users = user_counts[user_counts >= min_user_interactions].index
        filtered = filtered[filtered["user"].isin(kept_users)]

        item_counts = filtered.groupby("item").size()
        kept_items = item_counts[item_counts >= min_item_interactions].index
        filtered = filtered[filtered["item"].isin(kept_items)]
        if len(filtered) == before:
            break

    return filtered.drop_duplicates(["user", "item"]).sort_values(["user", "ts", "item"]).reset_index(drop=True)


def reindex_ciao(
    ratings: pd.DataFrame,
    trust: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int], dict[int, int]]:
    required = {"user", "friend"}
    missing = required.difference(trust.columns)
    if missing:
        raise ValueError(f"trust missing required columns: {sorted(missing)}")

    users = sorted(ratings["user"].unique().tolist())
    items = sorted(ratings["item"].unique().tolist())
    user_map = {int(user): idx for idx, user in enumerate(users)}
    item_map = {int(item): idx for idx, item in enumerate(items)}

    record = pd.DataFrame(
        {
            "uid": ratings["user"].map(user_map).astype(np.int64),
            "iid": ratings["item"].map(item_map).astype(np.int64),
            "ts": ratings["ts"].astype(np.int64),
        }
    )[BASE_COLUMNS]

    trust = trust[trust["user"].isin(user_map) & trust["friend"].isin(user_map)].copy()
    social_edges = pd.DataFrame(
        {
            "src": trust["user"].map(user_map).astype(np.int64),
            "dst": trust["friend"].map(user_map).astype(np.int64),
        }
    ).drop_duplicates()
    social_edges = social_edges[social_edges["src"] != social_edges["dst"]].reset_index(drop=True)
    return record, social_edges, user_map, item_map


def build_social_adj(social_edges: pd.DataFrame, n_user: int) -> sp.coo_matrix:
    if social_edges.empty:
        return sp.coo_matrix((n_user, n_user), dtype=np.float32)
    src = social_edges["src"].to_numpy(dtype=np.int64)
    dst = social_edges["dst"].to_numpy(dtype=np.int64)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    data = np.ones(len(row), dtype=np.float32)
    return sp.coo_matrix((data, (row, col)), shape=(n_user, n_user)).tocsr().sign().tocoo()


def _split_records(
    record: pd.DataFrame,
    n_item: int,
    test_frac: float,
    val_frac: float,
    skew_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    popularity = compute_popularity(record, n_item)
    test_record = popularity_weighted_user_sample(
        record,
        popularity=popularity,
        frac=test_frac,
        mode="uniform",
        cap_percentile=None,
        seed=seed,
    )
    train_val_pool = subtract_records(record, test_record)
    train_candidate, val_record = temporal_validation_split(train_val_pool, val_frac=val_frac)
    train_popularity = compute_popularity(train_candidate, n_item)
    train_skew_record = popularity_weighted_user_sample(
        train_candidate,
        popularity=train_popularity,
        frac=skew_frac,
        mode="direct",
        cap_percentile=90.0,
        seed=seed + 1,
    )
    train_record = subtract_records(train_candidate, train_skew_record)
    return train_record, train_skew_record, val_record, test_record


def convert_raw_ciao(
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
) -> dict[str, int | float]:
    ratings = read_ciao_ratings(rating_path)
    trust = read_ciao_trust(trust_path)
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
    (output_dir / "ciao_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rating-path", type=Path, required=True)
    parser.add_argument("--trust-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ciao/output"))
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--skew-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    summary = convert_raw_ciao(
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
