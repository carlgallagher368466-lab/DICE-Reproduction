from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config.const as const_util
from utils import ContextManager


def test_epinions_dataset_sets_load_path():
    flags = SimpleNamespace(dataset="epinions", load_path="")

    ContextManager.set_load_path(flags)

    assert flags.load_path == const_util.epinions
