import os
import sys

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import candidate_generator as cg


class DotScorer(torch.nn.Module):

    def __init__(self):
        super(DotScorer, self).__init__()
        self.user = torch.nn.Embedding(2, 3)
        self.item = torch.nn.Embedding(4, 3)
        with torch.no_grad():
            self.user.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
            self.item.weight.copy_(torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ]))

    def score(self, users, items):
        return (self.user(users) * self.item(items)).sum(-1)


def test_torch_scoring_topk_generator_returns_expected_items():
    scorer = DotScorer()
    generator = cg.TorchScoringTopKGenerator(scorer, num_items=4, device=torch.device("cpu"), item_chunk_size=2)

    result = generator.generate(np.array([0, 1]), k=2)

    assert result.tolist() == [[1, 2], [0, 3]]
