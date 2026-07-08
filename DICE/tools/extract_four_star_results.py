import argparse
import csv
import re
from pathlib import Path


RESULT_RE = re.compile(r"TEST results topk = (\d+):")
METRIC_RE = re.compile(r"(recall|hit_ratio|ndcg): ([0-9.]+)")
BEST_EPOCH_RE = re.compile(r"best epoch:\s*(\d+)")
SPLIT_RE = re.compile(r"ml10m-(longtail|uniform|head)-", re.IGNORECASE)

FIELDNAMES = [
    "split",
    "model",
    "best_epoch",
    "recall@20",
    "ndcg@20",
    "hit@20",
    "recall@50",
    "ndcg@50",
    "hit@50",
]

SPLIT_ORDER = {"longtail": 0, "uniform": 1, "head": 2}
MODEL_ORDER = {"MF": 0, "IPS": 1, "DICE": 2, "NeuMF": 3, "NeuMFIPS": 4, "NeuMFDICE": 5}


def parse_log(path):
    results = {}
    best_epoch = None
    current_topk = None
    for line in Path(path).read_text(errors="replace").splitlines():
        best_match = BEST_EPOCH_RE.search(line)
        if best_match:
            best_epoch = int(best_match.group(1))
            continue

        topk_match = RESULT_RE.search(line)
        if topk_match:
            current_topk = int(topk_match.group(1))
            results[current_topk] = {}
            continue

        metric_match = METRIC_RE.search(line)
        if metric_match and current_topk is not None:
            metric, value = metric_match.groups()
            if metric == "hit_ratio":
                metric = "hit"
            results[current_topk][metric] = float(value)

    return best_epoch, results


def infer_split_model(path):
    text = Path(path).as_posix()
    lower = text.lower()
    split_match = SPLIT_RE.search(text)
    if not split_match:
        raise ValueError(f"Cannot infer split from {path}")
    split = split_match.group(1).lower()

    if "neumfdice" in lower:
        model = "NeuMFDICE"
    elif "neumfips" in lower:
        model = "NeuMFIPS"
    elif "neumf" in lower:
        model = "NeuMF"
    elif "dice" in lower:
        model = "DICE"
    elif "ips" in lower:
        model = "IPS"
    else:
        model = "MF"
    return split, model


def record_from_log(path):
    split, model = infer_split_model(path)
    best_epoch, parsed = parse_log(path)
    if 20 not in parsed or 50 not in parsed:
        return None

    return {
        "split": split,
        "model": model,
        "best_epoch": best_epoch,
        "recall@20": parsed[20].get("recall"),
        "ndcg@20": parsed[20].get("ndcg"),
        "hit@20": parsed[20].get("hit"),
        "recall@50": parsed[50].get("recall"),
        "ndcg@50": parsed[50].get("ndcg"),
        "hit@50": parsed[50].get("hit"),
    }


def collect_records(root):
    root = Path(root)
    records = []
    for path in sorted(root.glob("output/*/test_log/*")):
        if not path.is_file():
            continue
        record = record_from_log(path)
        if record:
            records.append(record)
    records.sort(key=lambda r: (SPLIT_ORDER[r["split"]], MODEL_ORDER.get(r["model"], 99)))
    return records


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_outputs(records, csv_path, markdown_path):
    if not records:
        raise ValueError("No records to write.")

    csv_path = Path(csv_path)
    markdown_path = Path(markdown_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "| Split | Model | Best Epoch | Recall@20 | NDCG@20 | Hit@20 | Recall@50 | NDCG@50 | Hit@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(format_value(record[field]) for field in FIELDNAMES)
            + " |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Run directory containing output/*/test_log.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--md", required=True)
    args = parser.parse_args()

    records = collect_records(args.root)
    write_outputs(records, args.csv, args.md)


if __name__ == "__main__":
    main()
