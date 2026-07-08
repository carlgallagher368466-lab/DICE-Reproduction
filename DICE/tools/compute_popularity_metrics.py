import argparse
import csv
import re
from pathlib import Path

import numpy as np


FIELDNAMES = [
    "split",
    "model",
    "topk",
    "avg_pop",
    "longtail_ratio",
    "coverage",
    "num_users",
]

SPLIT_RE = re.compile(r"(longtail|uniform|head)", re.IGNORECASE)
MODEL_RE = re.compile(r"(NeuMFDICE|NeuMFIPS|NeuMF|DICE|IPS|MF)", re.IGNORECASE)


def read_recommendations(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return np.empty((0, 0), dtype=np.int64)

        has_user_id = header[0].lower() in {"user_id", "user", "uid"}
        for row in reader:
            if not row:
                continue
            values = row[1:] if has_user_id else row
            rows.append([int(value) for value in values if value != ""])

    if not rows:
        return np.empty((0, 0), dtype=np.int64)
    width = min(len(row) for row in rows)
    return np.asarray([row[:width] for row in rows], dtype=np.int64)


def infer_split_model(path):
    name = Path(path).as_posix()
    split_match = SPLIT_RE.search(name)
    model_match = MODEL_RE.search(name)
    if not split_match or not model_match:
        raise ValueError(f"Cannot infer split/model from {path}")

    model_token = model_match.group(1).lower()
    model_map = {
        "neumfdice": "NeuMFDICE",
        "neumfips": "NeuMFIPS",
        "neumf": "NeuMF",
        "dice": "DICE",
        "ips": "IPS",
        "mf": "MF",
    }
    return split_match.group(1).lower(), model_map[model_token]


def compute_file_metrics(
    recommendation_path,
    popularity,
    topks,
    longtail_percentile=20.0,
    split=None,
    model=None,
):
    recommendations = read_recommendations(recommendation_path)
    if recommendations.size == 0:
        return []

    if split is None or model is None:
        inferred_split, inferred_model = infer_split_model(recommendation_path)
        split = split or inferred_split
        model = model or inferred_model

    popularity = np.asarray(popularity, dtype=np.float64)
    threshold = np.percentile(popularity, longtail_percentile)
    records = []
    for topk in topks:
        actual_k = min(int(topk), recommendations.shape[1])
        recs = recommendations[:, :actual_k]
        rec_pop = popularity[recs]
        longtail_mask = rec_pop <= threshold
        unique_items = np.unique(recs)

        records.append(
            {
                "split": split,
                "model": model,
                "topk": actual_k,
                "avg_pop": float(rec_pop.mean()),
                "longtail_ratio": float(longtail_mask.mean()),
                "coverage": float(len(unique_items) / len(popularity)),
                "num_users": int(recommendations.shape[0]),
            }
        )
    return records


def collect_records(recommendation_root, popularity_path, topks, longtail_percentile=20.0):
    popularity = np.load(popularity_path)
    records = []
    for path in sorted(Path(recommendation_root).glob("*.csv")):
        records.extend(
            compute_file_metrics(
                path,
                popularity,
                topks=topks,
                longtail_percentile=longtail_percentile,
            )
        )
    return records


def write_csv(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommendation-root", required=True)
    parser.add_argument("--popularity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, action="append", default=[20, 50])
    parser.add_argument("--longtail-percentile", type=float, default=20.0)
    args = parser.parse_args()

    records = collect_records(
        args.recommendation_root,
        args.popularity,
        topks=args.topk,
        longtail_percentile=args.longtail_percentile,
    )
    write_csv(records, args.output)


if __name__ == "__main__":
    main()
