import os
import sys

import numpy as np
import scipy.sparse as sp


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import utils


class Flags:
    name = "test"


def test_dice_sampler_falls_back_when_popularity_margin_has_no_candidates():
    lil = sp.coo_matrix(
        (
            np.ones(2, dtype=np.float32),
            ([0, 0], [0, 1]),
        ),
        shape=(1, 4),
    ).tolil()
    dok = lil.todok()
    popularity = np.array([5, 5, 5, 5], dtype=np.float32)

    sampler = utils.DICESampler(
        Flags(),
        lil_record=lil,
        dok_record=dok,
        neg_sample_rate=4,
        popularity=popularity,
        margin=40,
        pool=40,
    )

    negatives, mask = sampler.generate_negative_samples(user=0, pos_item=0)

    assert negatives.shape == (4,)
    assert mask.shape == (4,)
    assert set(negatives).issubset({2, 3})
