#!/usr/local/anaconda3/envs/torch-1.1-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error


ml10m = '../data/ml10m/output/'
nf = '../data/netflix/output/'
ml10m_longtail_test = '../data/three_star/ml10m_longtail_test/output/'
ml10m_head_test = '../data/three_star/ml10m_head_test/output/'
ml10m_uniform_test = '../data/three_star/ml10m_uniform_test/output/'
ciao = '../data/ciao/output/'
epinions = '../data/epinions/output/'

coo_record = 'coo_record.npz'
train_coo_record = 'train_coo_record.npz'
val_coo_record = 'val_coo_record.npz'
test_coo_record = 'test_coo_record.npz'

train_skew_coo_record = 'train_skew_coo_record.npz'

popularity = 'popularity.npy'
blend_popularity = 'popularity_blend.npy'
train_coo_adj_graph = 'train_coo_adj_graph.npz'
train_skew_coo_adj_graph = 'train_skew_coo_adj_graph.npz'
train_blend_coo_adj_graph = 'train_blend_coo_adj_graph.npz'
social_edges = 'social_edges.csv'
social_adj = 'social_adj.npz'

ckpt = 'ckpt/'
