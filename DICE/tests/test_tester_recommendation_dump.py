import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

for module_name in ["faiss", "visdom", "setproctitle", "dgl"]:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

fake_metrics = types.ModuleType("metrics")


class FakeJudger:
    def __init__(self, flags, dm, topk):
        self.metrics = flags.metrics

    def judge(self, items, test_pos, num_test_pos):
        return {}, len(items)


fake_metrics.Judger = FakeJudger
sys.modules["metrics"] = fake_metrics
sys.modules["utils"] = types.ModuleType("utils")
sys.modules["data"] = types.ModuleType("data")
sys.modules["recommender"] = types.ModuleType("recommender")

import tester


class FakeRecommender:
    def __init__(self):
        self.dm = SimpleNamespace(n_user=2)

    def make_cg(self):
        pass

    def cg(self, users, topk):
        return np.array([[3, 2, 1], [4, 2, 0]])[:, :topk]


def test_tester_dumps_filtered_recommendations(tmp_path):
    flags = SimpleNamespace(
        name="longtail-NeuMFDICE",
        topk=[2],
        batch_size=2,
        metrics=[],
        dump_recommendations_dir=str(tmp_path),
        workspace=str(tmp_path),
    )
    t = tester.Tester(flags, FakeRecommender())
    t.dataloader = [
        (
            torch.tensor([0, 1]),
            [[3], [0]],
            [np.array([2]), np.array([4])],
            torch.tensor([1, 1]),
        )
    ]
    t.topk_margin = 1
    t.cg_topk = 3
    t.n_user = 2
    t.test_data_source = "test"

    t.test(num_test_users=2)

    dump_path = tmp_path / "longtail-NeuMFDICE_top2.csv"
    assert dump_path.read_text().splitlines() == [
        "user_id,item_1,item_2",
        "0,2,1",
        "1,4,2",
    ]
