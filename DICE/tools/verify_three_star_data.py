from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


REQUIRED_FILES = [
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
    "three_star_summary.json",
]


def verify_variant(output_dir: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{output_dir} missing files: {missing}")

    train = sp.load_npz(output_dir / "train_coo_record.npz")
    train_skew = sp.load_npz(output_dir / "train_skew_coo_record.npz")
    val = sp.load_npz(output_dir / "val_coo_record.npz")
    test = sp.load_npz(output_dir / "test_coo_record.npz")
    for name, matrix in [("train_skew", train_skew), ("val", val), ("test", test)]:
        if matrix.shape != train.shape:
            raise ValueError(f"{output_dir} {name} shape {matrix.shape} != train shape {train.shape}")

    n_user, n_item = train.shape
    for name in ["train_coo_adj_graph.npz", "train_skew_coo_adj_graph.npz", "train_blend_coo_adj_graph.npz"]:
        adj = sp.load_npz(output_dir / name)
        if adj.shape != (n_user + n_item, n_user + n_item):
            raise ValueError(f"{output_dir} {name} has invalid shape {adj.shape}")

    for name in ["popularity.npy", "popularity_all.npy", "popularity_skew.npy", "popularity_blend.npy"]:
        popularity = np.load(output_dir / name)
        if popularity.shape != (n_item,):
            raise ValueError(f"{output_dir} {name} has invalid shape {popularity.shape}")

    summary = json.loads((output_dir / "three_star_summary.json").read_text(encoding="utf-8"))
    return {
        "variant": output_dir.parent.name,
        "shape": train.shape,
        "train": train.nnz,
        "train_skew": train_skew.nnz,
        "val": val.nnz,
        "test": test.nnz,
        "test_mean_item_popularity": round(summary["test_mean_item_popularity"], 2),
        "train_mean_item_popularity": round(summary["train_mean_item_popularity"], 2),
    }


def main() -> None:
    root = Path("data/three_star")
    outputs = sorted(root.glob("ml10m_*_test/output"))
    if not outputs:
        raise FileNotFoundError("No three-star outputs found under data/three_star")

    for output_dir in outputs:
        result = verify_variant(output_dir)
        print(result)


if __name__ == "__main__":
    main()
