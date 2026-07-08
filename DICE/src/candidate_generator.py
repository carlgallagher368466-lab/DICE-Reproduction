#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from collections.abc import Iterable

import numpy as np
import torch

import faiss


class CandidateGenerator(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_cg'

    def generate(self, user, k):

        raise NotImplementedError


class RandomGenerator(CandidateGenerator):

    def __init__(self, flags_obj, items):

        super(RandomGenerator, self).__init__(flags_obj)
        self.items = items
        self.n_item = len(items)

    def generate(self, users, k):

        if not isinstance(users, Iterable):
            return self.choice(k)
        items_chosen = [self.choice(k) for _ in users]
        return np.stack(items_chosen, axis=0)

    def choice(self, k):

        item_chosen = np.full(k, -1)
        for count in range(k):
            i = np.random.randint(self.n_item)
            while i in item_chosen:
                i = np.random.randint(self.n_item)
            item_chosen[count] = i
        return item_chosen


class PopularityGenerator(CandidateGenerator):

    def __init__(self, flags_obj, popularity, max_k):

        super(PopularityGenerator, self).__init__(flags_obj)
        self.popularity = popularity
        self.max_k = max_k
        self.get_popular_items()

    def get_popular_items(self):

        popularity_tensor = torch.LongTensor(self.popularity)
        self.popular_items = torch.topk(popularity_tensor, self.max_k)[1].numpy()

    def generate(self, user, k):

        if not isinstance(user, Iterable):
            return self.popular_items[:k]
        items = [self.popular_items[:k] for _ in user]
        return np.stack(items, axis=0)


class FaissInnerProductMaximumSearchGenerator(CandidateGenerator):

    def __init__(self, flags_obj, items):

        super(FaissInnerProductMaximumSearchGenerator, self).__init__(flags_obj)
        self.items = items
        self.embedding_size = items.shape[1]
        self.make_index(flags_obj)

    def make_index(self, flags_obj):

        self.make_index_brute_force(flags_obj)

        if flags_obj.cg_use_gpu:

            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, flags_obj.cg_gpu_id, self.index)

    def make_index_brute_force(self, flags_obj):

        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.index.add(self.items)

    def generate(self, users, k):

        _, I = self.index.search(users, k)

        return I

    def generate_with_distance(self, users, k):

        D, I = self.index.search(users, k)

        return D, I


class TorchScoringTopKGenerator(CandidateGenerator):

    def __init__(self, model, num_items, device, item_chunk_size=2048):

        self.name = 'torch_scoring_topk_cg'
        self.model = model
        self.num_items = num_items
        self.device = device
        self.item_chunk_size = item_chunk_size

    def generate(self, users, k):

        if not isinstance(users, Iterable):
            users = np.array([users])
        users = np.asarray(users, dtype=np.int64)

        user_tensor = torch.LongTensor(users).to(self.device)
        top_scores = None
        top_items = None

        for start in range(0, self.num_items, self.item_chunk_size):
            end = min(start + self.item_chunk_size, self.num_items)
            item_tensor = torch.arange(start, end, dtype=torch.long, device=self.device)
            users_expanded = user_tensor[:, None].expand(-1, end - start)
            items_expanded = item_tensor[None, :].expand(len(users), -1)
            scores = self.model.score(users_expanded, items_expanded)

            chunk_k = min(k, end - start)
            chunk_scores, chunk_items = torch.topk(scores, chunk_k, dim=1)
            chunk_items = chunk_items + start

            if top_scores is None:
                top_scores = chunk_scores
                top_items = chunk_items
            else:
                merged_scores = torch.cat((top_scores, chunk_scores), dim=1)
                merged_items = torch.cat((top_items, chunk_items), dim=1)
                merged_k = min(k, merged_scores.size(1))
                top_scores, indices = torch.topk(merged_scores, merged_k, dim=1)
                top_items = torch.gather(merged_items, 1, indices)

        return top_items.detach().cpu().numpy()
