#!/usr/local/anaconda3/envs/torch-1.1-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import flags

import utils
import data
import metrics
import recommender as rec

import csv
import numpy as np
import os
import torch

from tqdm import tqdm
import time


class Tester(object):

    def __init__(self, flags_obj, recommender, test_iou=False):

        self.name = flags_obj.name + '_tester'
        self.recommender = recommender
        self.flags_obj = flags_obj
        self.set_topk(flags_obj)
        self.set_judger(flags_obj, recommender.dm)
        self.test_iou = test_iou
        if test_iou:
            self.pop_recommender = rec.PopularityRecommender(flags_obj, recommender.workspace, recommender.dm)

    def set_metrics(self, metrics):

        self.judger.metrics = metrics

    def set_dataloader(self, test_data_source):

        self.test_data_source = test_data_source
        self.dataloader, self.topk_margin = data.CGDataProcessor.get_dataloader(self.flags_obj, self.test_data_source)
        self.n_user = self.recommender.dm.n_user
        self.cg_topk = self.max_topk + self.topk_margin

    def set_topk(self, flags_obj):

        self.topk = flags_obj.topk
        self.max_topk = max(flags_obj.topk)

    def set_judger(self, flags_obj, dm):

        self.judger = metrics.Judger(flags_obj, dm, self.max_topk)

    def test(self, num_test_users):

        real_num_test_users = 0

        with torch.no_grad():

            self.init_results()
            self.init_recommendation_dump()
            self.make_cg()
            if self.test_iou:
                self.pop_recommender.make_cg()

            num_test_batches = num_test_users // self.flags_obj.batch_size + 1

            for batch_count, data in enumerate(tqdm(self.dataloader)):

                if batch_count >= num_test_batches:
                    break

                users, train_pos, test_pos, num_test_pos = data
                users = users.squeeze()

                items = self.recommender.cg(users, self.cg_topk)

                items = self.filter_history(items, train_pos)
                self.dump_recommendation_batch(users, items)

                batch_results, valid_num_users = self.judger.judge(items, test_pos, num_test_pos)

                real_num_test_users = real_num_test_users + valid_num_users

                if self.test_iou:
                    pop_items = self.pop_recommender.cg(users, self.cg_topk)
                    pop_items = self.filter_history(pop_items, train_pos)
                    iou = self.compute_iou(items, pop_items)
                    batch_results['iou'] = iou

                self.update_results(batch_results)

        self.average_user(real_num_test_users)
        self.close_recommendation_dump()

        return self.results

    def init_results(self):

        self.results = {k: 0.0 for k in self.judger.metrics}
        if self.test_iou:
            self.results['iou'] = 0.0

    def make_cg(self):

        self.recommender.make_cg()

    def init_recommendation_dump(self):

        dump_dir = getattr(self.flags_obj, 'dump_recommendations_dir', '')
        self.recommendation_dump_file = None
        self.recommendation_dump_writer = None
        if not dump_dir or self.test_data_source != 'test':
            return

        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, '{}_top{}.csv'.format(self.flags_obj.name, self.max_topk))
        self.recommendation_dump_file = open(path, 'w', newline='')
        self.recommendation_dump_writer = csv.writer(self.recommendation_dump_file)
        header = ['user_id'] + ['item_{}'.format(i + 1) for i in range(self.max_topk)]
        self.recommendation_dump_writer.writerow(header)

    def dump_recommendation_batch(self, users, items):

        if self.recommendation_dump_writer is None:
            return

        if torch.is_tensor(users):
            users = users.detach().cpu().numpy()
        users = np.asarray(users).reshape(-1)
        for user, user_items in zip(users, items):
            self.recommendation_dump_writer.writerow([int(user)] + [int(item) for item in user_items[:self.max_topk]])

    def close_recommendation_dump(self):

        if self.recommendation_dump_file is not None:
            self.recommendation_dump_file.close()
            self.recommendation_dump_file = None
            self.recommendation_dump_writer = None

    def filter_history(self, items, train_pos):

        return np.stack([items[i][np.isin(items[i], train_pos[i], invert=True)][:self.max_topk] for i in range(len(items))], axis=0)

    def compute_iou(self, items, pop_items):

        intersection = np.array([len(np.intersect1d(items[i], pop_items[i])) for i in range(len(items))])
        union = np.array([len(np.union1d(items[i], pop_items[i])) for i in range(len(items))])
        iou = np.sum(intersection/union)

        return iou

    def update_results(self, batch_results):

        for metric, value in batch_results.items():
            self.results[metric] = self.results[metric] + value

    def average_user(self, num_test_users):

        if num_test_users > self.n_user:
            num_test_users = self.n_user

        for metric, value in self.results.items():
            if metric in ['recall', 'hit_ratio', 'ndcg', 'iou']:
                self.results[metric] = value/num_test_users

