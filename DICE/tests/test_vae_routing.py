import os
import sys
import types


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.modules.setdefault("faiss", types.SimpleNamespace())

import recommender
import utils


class Flags(object):
    embedding_size = 4
    vae_latent_size = 3
    vae_hidden_size = 5
    vae_dropout = 0.0
    vae_kl_weight = 0.001
    dis_loss = "L2"
    dis_pen = 0.01
    int_weight = 0.1
    pop_weight = 0.1
    neumf_item_chunk_size = 16
    use_gpu = False
    model = "VAE"


class DataManager(object):
    n_user = 2
    n_item = 3

    def get_skew_dataset(self):
        self.skew_loaded = True


def test_context_manager_routes_vae_models():
    dm = DataManager()
    flags = Flags()

    flags.model = "VAE"
    assert isinstance(utils.ContextManager.set_recommender(flags, ".", dm), recommender.VAERecommender)

    flags.model = "VAEIPS"
    assert isinstance(utils.ContextManager.set_recommender(flags, ".", dm), recommender.VAEIPSRecommender)

    flags.model = "VAEDICE"
    assert isinstance(utils.ContextManager.set_recommender(flags, ".", dm), recommender.VAEDICERecommender)
