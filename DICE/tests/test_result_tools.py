import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_four_star_results_parses_test_logs(tmp_path):
    tool = load_tool("extract_four_star_results")
    log_dir = (
        tmp_path
        / "output"
        / "three-star-ml10m-longtail-NeuMFDICE_2026-06-16-16-37-35"
        / "test_log"
    )
    log_dir.mkdir(parents=True)
    (log_dir / "run.log.INFO.1").write_text(
        "\n".join(
            [
                "I trainer.py:74] best epoch: 14",
                "I trainer.py:84] TEST results topk = 20:",
                "I trainer.py:86] recall: 0.15505791690081075",
                "I trainer.py:86] hit_ratio: 0.534130859375",
                "I trainer.py:86] ndcg: 0.10379183999394137",
                "I trainer.py:84] TEST results topk = 50:",
                "I trainer.py:86] recall: 0.2812276515179162",
                "I trainer.py:86] hit_ratio: 0.739404296875",
                "I trainer.py:86] ndcg: 0.1447149214154617",
            ]
        )
    )

    records = tool.collect_records(tmp_path)

    assert records == [
        {
            "split": "longtail",
            "model": "NeuMFDICE",
            "best_epoch": 14,
            "recall@20": 0.15505791690081075,
            "ndcg@20": 0.10379183999394137,
            "hit@20": 0.534130859375,
            "recall@50": 0.2812276515179162,
            "ndcg@50": 0.1447149214154617,
            "hit@50": 0.739404296875,
        }
    ]


def test_extract_four_star_results_writes_csv_and_markdown(tmp_path):
    tool = load_tool("extract_four_star_results")
    records = [
        {
            "split": "head",
            "model": "NeuMF",
            "best_epoch": 19,
            "recall@20": 0.35483789130463195,
            "ndcg@20": 0.2429927604028856,
            "hit@20": 0.822119140625,
            "recall@50": 0.5726277190039333,
            "ndcg@50": 0.3168105252870969,
            "hit@50": 0.944482421875,
        }
    ]
    csv_path = tmp_path / "results.csv"
    md_path = tmp_path / "results.md"

    tool.write_outputs(records, csv_path, md_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["split"] == "head"
    assert rows[0]["model"] == "NeuMF"
    assert rows[0]["recall@20"] == "0.35483789130463195"
    assert "| head | NeuMF | 19 | 0.3548 |" in md_path.read_text()
