import csv
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools" / "compute_popularity_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_popularity_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_popularity_metrics_from_recommendation_csv(tmp_path):
    tool = load_tool()
    popularity = np.array([100, 80, 20, 5, 1], dtype=np.float32)
    pop_path = tmp_path / "popularity.npy"
    np.save(pop_path, popularity)
    rec_path = tmp_path / "longtail_NeuMFDICE.csv"
    rec_path.write_text("user_id,item_1,item_2,item_3\n0,4,3,2\n1,3,2,1\n")

    record = tool.compute_file_metrics(
        rec_path,
        popularity,
        topks=[2, 3],
        longtail_percentile=40.0,
        split="longtail",
        model="NeuMFDICE",
    )

    assert record == [
        {
            "split": "longtail",
            "model": "NeuMFDICE",
            "topk": 2,
            "avg_pop": 7.75,
            "longtail_ratio": 0.75,
            "coverage": 0.6,
            "num_users": 2,
        },
        {
            "split": "longtail",
            "model": "NeuMFDICE",
            "topk": 3,
            "avg_pop": 21.833333333333332,
            "longtail_ratio": 0.5,
            "coverage": 0.8,
            "num_users": 2,
        },
    ]


def test_compute_popularity_metrics_writes_csv(tmp_path):
    tool = load_tool()
    output = tmp_path / "metrics.csv"
    records = [
        {
            "split": "longtail",
            "model": "NeuMFDICE",
            "topk": 20,
            "avg_pop": 12.5,
            "longtail_ratio": 0.4,
            "coverage": 0.25,
            "num_users": 100,
        }
    ]

    tool.write_csv(records, output)

    with output.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["model"] == "NeuMFDICE"
    assert rows[0]["avg_pop"] == "12.5"
