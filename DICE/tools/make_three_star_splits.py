"""Build DICE-compatible data variants for three-star reproduction.

The variants focus on the rubric's train/test construction requirement:
sampling interactions with item-popularity weights while preserving DICE's
expected sparse matrix filenames and shapes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


BASE_COLUMNS = ["uid", "iid", "ts"]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    mode: str
    test_frac: float
    skew_frac: float
    cap_percentile: float | None
    seed: int


def read_record(path: Path) -> pd.DataFrame:
    record = pd.read_csv(path)
    missing = [column for column in BASE_COLUMNS if column not in record.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return record[BASE_COLUMNS].drop_duplicates().reset_index(drop=True)


def infer_shape(output_dir: Path, full_record: pd.DataFrame) -> tuple[int, int]:
    train_coo_path = output_dir / "train_coo_record.npz"
    if train_coo_path.exists():
        coo = sp.load_npz(train_coo_path)
        return coo.shape
    return int(full_record["uid"].max()) + 1, int(full_record["iid"].max()) + 1


def compute_popularity(record: pd.DataFrame, n_item: int) -> np.ndarray:
    popularity = np.zeros(n_item, dtype=np.float32)
    counts = record.groupby("iid").size()
    popularity[counts.index.to_numpy(dtype=np.int64)] = counts.to_numpy(dtype=np.float32)
    return popularity


def _weights_for(record: pd.DataFrame, popularity: np.ndarray, mode: str) -> np.ndarray:
    item_pop = popularity[record["iid"].to_numpy(dtype=np.int64)].astype(np.float64)
    item_pop = np.clip(item_pop, 1.0, None)

    if mode == "inverse":
        weights = 1.0 / item_pop
    elif mode == "direct":
        weights = item_pop
    elif mode == "uniform":
        weights = np.ones_like(item_pop, dtype=np.float64)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    return weights / weights.sum()


def _apply_cap(weights: np.ndarray, cap_percentile: float | None) -> np.ndarray:
    if cap_percentile is None:
        return weights
    capped = np.minimum(weights, np.percentile(weights, cap_percentile))
    return capped / capped.sum()


def popularity_weighted_user_sample(
    record: pd.DataFrame,
    popularity: np.ndarray,
    frac: float,
    mode: str,
    cap_percentile: float | None,
    seed: int,
) -> pd.DataFrame:
    if not 0 < frac < 1:
        raise ValueError("frac must be between 0 and 1")

    sampled_parts = []
    rng = np.random.default_rng(seed)
    for _, user_record in record.groupby("uid", sort=False):
        n_sample = max(1, int(round(len(user_record) * frac)))
        if n_sample >= len(user_record):
            n_sample = max(1, len(user_record) - 1)
        weights = _weights_for(user_record, popularity, mode)
        weights = _apply_cap(weights, cap_percentile)
        picked = rng.choice(user_record.index.to_numpy(), size=n_sample, replace=False, p=weights)
        sampled_parts.append(user_record.loc[picked, BASE_COLUMNS])

    return pd.concat(sampled_parts, ignore_index=True).drop_duplicates().reset_index(drop=True)


def subtract_records(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.merge(right[BASE_COLUMNS].drop_duplicates(), how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"][BASE_COLUMNS].reset_index(drop=True)


def temporal_validation_split(record: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts_train = []
    parts_val = []
    ranked = record.sort_values(["uid", "ts"]).reset_index(drop=True)
    for _, user_record in ranked.groupby("uid", sort=False):
        n_val = max(1, int(round(len(user_record) * val_frac)))
        n_val = min(n_val, max(0, len(user_record) - 1))
        if n_val == 0:
            parts_train.append(user_record)
            continue
        parts_val.append(user_record.tail(n_val))
        parts_train.append(user_record.iloc[:-n_val])
    train = pd.concat(parts_train, ignore_index=True)[BASE_COLUMNS]
    val = pd.concat(parts_val, ignore_index=True)[BASE_COLUMNS] if parts_val else record.iloc[0:0][BASE_COLUMNS]
    return train.reset_index(drop=True), val.reset_index(drop=True)


def build_coo(record: pd.DataFrame, n_user: int, n_item: int) -> sp.coo_matrix:
    row = record["uid"].to_numpy(dtype=np.int64)
    col = record["iid"].to_numpy(dtype=np.int64)
    data = np.ones(len(record), dtype=np.float32)
    return sp.coo_matrix((data, (row, col)), shape=(n_user, n_item))


def build_bipartite_adj(coo: sp.coo_matrix) -> sp.coo_matrix:
    coo = coo.tocoo()
    n_user, n_item = coo.shape
    row = np.concatenate([coo.row, coo.col + n_user])
    col = np.concatenate([coo.col + n_user, coo.row])
    data = np.ones(len(row), dtype=np.float32)
    return sp.coo_matrix((data, (row, col)), shape=(n_user + n_item, n_user + n_item))


def write_csv(path: Path, record: pd.DataFrame) -> None:
    record[BASE_COLUMNS].reset_index(drop=True).to_csv(path)


def write_variant(
    output_dir: Path,
    full_record: pd.DataFrame,
    train_record: pd.DataFrame,
    train_skew_record: pd.DataFrame,
    val_record: pd.DataFrame,
    test_record: pd.DataFrame,
    n_user: int,
    n_item: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    records = {
        "record": full_record,
        "train": train_record,
        "train_skew": train_skew_record,
        "val": val_record,
        "test": test_record,
    }
    for name, record in records.items():
        write_csv(output_dir / f"{name}_record.csv" if name != "record" else output_dir / "record.csv", record)

    coos = {
        "coo_record": build_coo(full_record, n_user, n_item),
        "train_coo_record": build_coo(train_record, n_user, n_item),
        "train_skew_coo_record": build_coo(train_skew_record, n_user, n_item),
        "val_coo_record": build_coo(val_record, n_user, n_item),
        "test_coo_record": build_coo(test_record, n_user, n_item),
    }
    for name, coo in coos.items():
        sp.save_npz(output_dir / f"{name}.npz", coo)

    train_blend = (coos["train_coo_record"] + coos["train_skew_coo_record"]).tocoo()
    sp.save_npz(output_dir / "train_coo_adj_graph.npz", build_bipartite_adj(coos["train_coo_record"]))
    sp.save_npz(output_dir / "train_skew_coo_adj_graph.npz", build_bipartite_adj(coos["train_skew_coo_record"]))
    sp.save_npz(output_dir / "train_blend_coo_adj_graph.npz", build_bipartite_adj(train_blend))

    popularity = compute_popularity(train_record, n_item)
    popularity_skew = compute_popularity(train_skew_record, n_item)
    popularity_all = compute_popularity(full_record, n_item)
    popularity_blend = popularity + popularity_skew
    np.save(output_dir / "popularity.npy", popularity)
    np.save(output_dir / "popularity_skew.npy", popularity_skew)
    np.save(output_dir / "popularity_all.npy", popularity_all)
    np.save(output_dir / "popularity_blend.npy", popularity_blend)


def copy_index_files(source_dir: Path, output_dir: Path) -> None:
    for filename in ["user_reindex.json", "item_reindex.json"]:
        source = source_dir / filename
        if source.exists():
            (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def build_variant(source_dir: Path, destination_root: Path, spec: VariantSpec, val_frac: float) -> Path:
    full_record = read_record(source_dir / "record.csv")
    n_user, n_item = infer_shape(source_dir, full_record)
    popularity = compute_popularity(full_record, n_item)

    test_record = popularity_weighted_user_sample(
        full_record,
        popularity=popularity,
        frac=spec.test_frac,
        mode=spec.mode,
        cap_percentile=spec.cap_percentile,
        seed=spec.seed,
    )
    train_val_pool = subtract_records(full_record, test_record)
    train_candidate, val_record = temporal_validation_split(train_val_pool, val_frac=val_frac)
    train_popularity = compute_popularity(train_candidate, n_item)
    train_skew_record = popularity_weighted_user_sample(
        train_candidate,
        popularity=train_popularity,
        frac=spec.skew_frac,
        mode=spec.mode,
        cap_percentile=spec.cap_percentile,
        seed=spec.seed + 1,
    )
    train_record = subtract_records(train_candidate, train_skew_record)

    output_dir = destination_root / spec.name / "output"
    write_variant(output_dir, full_record, train_record, train_skew_record, val_record, test_record, n_user, n_item)
    copy_index_files(source_dir, output_dir)

    summary = {
        "name": spec.name,
        "mode": spec.mode,
        "test_frac": spec.test_frac,
        "skew_frac": spec.skew_frac,
        "cap_percentile": spec.cap_percentile,
        "seed": spec.seed,
        "n_user": n_user,
        "n_item": n_item,
        "records": {
            "full": int(len(full_record)),
            "train": int(len(train_record)),
            "train_skew": int(len(train_skew_record)),
            "val": int(len(val_record)),
            "test": int(len(test_record)),
        },
        "test_mean_item_popularity": float(popularity[test_record["iid"]].mean()),
        "train_mean_item_popularity": float(popularity[train_record["iid"]].mean()),
        "skew_mean_item_popularity": float(popularity[train_skew_record["iid"]].mean()),
    }
    (output_dir / "three_star_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir


def parse_specs(args: argparse.Namespace) -> list[VariantSpec]:
    specs = [
        VariantSpec(
            name=f"{args.dataset}_{name}",
            mode=mode,
            test_frac=args.test_frac,
            skew_frac=args.skew_frac,
            cap_percentile=args.cap_percentile,
            seed=args.seed + idx * 100,
        )
        for idx, (name, mode) in enumerate(
            [
                ("longtail_test", "inverse"),
                ("head_test", "direct"),
                ("uniform_test", "uniform"),
            ]
        )
    ]
    if args.variant != "all":
        specs = [spec for spec in specs if spec.name.endswith(args.variant)]
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml10m", "netflix"], default="ml10m")
    parser.add_argument("--variant", choices=["all", "longtail_test", "head_test", "uniform_test"], default="all")
    parser.add_argument("--source-root", type=Path, default=Path("data"))
    parser.add_argument("--destination-root", type=Path, default=Path("data/three_star"))
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--skew-frac", type=float, default=0.2)
    parser.add_argument("--cap-percentile", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source_name = "netflix" if args.dataset == "netflix" else args.dataset
    source_dir = args.source_root / source_name / "output"
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    for spec in parse_specs(args):
        output_dir = build_variant(source_dir, args.destination_root, spec, val_frac=args.val_frac)
        print(output_dir)


if __name__ == "__main__":
    main()
