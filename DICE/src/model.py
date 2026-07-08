#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

try:
    import dgl.function as fn
except ImportError:
    fn = None

import utils

from deprecated import deprecated
from tqdm import tqdm

import random


class MF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(MF, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)

    def pair_forward(self, user, item_p, item_n):

        user = self.users[user]
        item_p = self.items[item_p]
        item_n = self.items[item_n]

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def point_forward(self, user, item):

        user = self.users[user]
        item = self.items[item]

        score = torch.sum(user * item, 2)

        return score

    def get_item_embeddings(self):

        return self.items.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        return self.users.detach().cpu().numpy().astype('float32')


class NeuMF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, mlp_layers, dropout):

        super(NeuMF, self).__init__()

        self.user_gmf = nn.Embedding(num_users, embedding_size)
        self.item_gmf = nn.Embedding(num_items, embedding_size)
        self.user_mlp = nn.Embedding(num_users, embedding_size)
        self.item_mlp = nn.Embedding(num_items, embedding_size)

        layers = []
        input_size = embedding_size * 2
        for output_size in mlp_layers:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = output_size
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(embedding_size + input_size, 1)

        self.init_params()

    def init_params(self):

        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)
        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def score(self, user, item):

        user_gmf = self.user_gmf(user)
        item_gmf = self.item_gmf(item)
        gmf = user_gmf * item_gmf

        user_mlp = self.user_mlp(user)
        item_mlp = self.item_mlp(item)
        mlp_input = torch.cat((user_mlp, item_mlp), dim=-1)
        mlp_out = self.mlp(mlp_input)

        score_input = torch.cat((gmf, mlp_out), dim=-1)
        return self.output(score_input).squeeze(-1)

    def pair_forward(self, user, item_p, item_n):

        p_score = self.score(user, item_p)
        n_score = self.score(user, item_n)

        return p_score, n_score


class NeuMFDICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, mlp_layers, dropout, dis_loss, dis_pen, int_weight, pop_weight):

        super(NeuMFDICE, self).__init__()

        self.interest = NeuMF(num_users, num_items, embedding_size, mlp_layers, dropout)
        self.popularity = NeuMF(num_users, num_items, embedding_size, mlp_layers, dropout)

        self.int_weight = int_weight
        self.pop_weight = pop_weight
        self.dis_pen = dis_pen

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

    def adapt(self, epoch, decay):

        self.int_weight = self.int_weight*decay
        self.pop_weight = self.pop_weight*decay

    def dcor(self, x, y):

        a = torch.norm(x[:,None] - x, p = 2, dim = 2)
        b = torch.norm(y[:,None] - y, p = 2, dim = 2)

        A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
        B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask*torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask):

        p_score_int, n_score_int = self.interest.pair_forward(user, item_p, item_n)
        p_score_pop, n_score_pop = self.popularity.pair_forward(user, item_p, item_n)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = self.interest.item_gmf(item_all)
        item_pop = self.popularity.item_gmf(item_all)
        user_all = torch.unique(user)
        user_int = self.interest.user_gmf(user_all)
        user_pop = self.popularity.user_gmf(user_all)
        discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

        loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total - self.dis_pen*discrepency_loss

        return loss

    def score(self, user, item):

        return self.interest.score(user, item) + self.popularity.score(user, item)

    def score_interest(self, user, item):

        return self.interest.score(user, item)

    def score_popularity(self, user, item):

        return self.popularity.score(user, item)


class VAE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, latent_size, hidden_size, dropout):

        super(VAE, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, latent_size)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

        self.init_params()

    def init_params(self):

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.xavier_uniform_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def encode(self, user, training=True):

        user_embedding = self.user_embedding(user)
        hidden = self.encoder(user_embedding)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(min=-10.0, max=10.0)
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return z, kl_loss

    def score(self, user, item):

        z, _ = self.encode(user, training=self.training)
        item_embedding = self.item_embedding(item)

        return torch.sum(z * item_embedding, dim=-1)

    def pair_forward(self, user, item_p, item_n):

        z, kl_loss = self.encode(user, training=self.training)
        item_p = self.item_embedding(item_p)
        item_n = self.item_embedding(item_n)

        p_score = torch.sum(z * item_p, dim=-1)
        n_score = torch.sum(z * item_n, dim=-1)

        return p_score, n_score, kl_loss

    def get_item_embeddings(self):

        return self.item_embedding.weight.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        z, _ = self.encode(torch.arange(self.user_embedding.num_embeddings, device=self.user_embedding.weight.device), training=False)
        return z.detach().cpu().numpy().astype('float32')


class VAEDICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, latent_size, hidden_size, dropout, dis_loss, dis_pen, int_weight, pop_weight, kl_weight):

        super(VAEDICE, self).__init__()

        self.interest = VAE(num_users, num_items, embedding_size, latent_size, hidden_size, dropout)
        self.popularity = VAE(num_users, num_items, embedding_size, latent_size, hidden_size, dropout)

        self.int_weight = int_weight
        self.pop_weight = pop_weight
        self.dis_pen = dis_pen
        self.kl_weight = kl_weight

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

    def adapt(self, epoch, decay):

        self.int_weight = self.int_weight*decay
        self.pop_weight = self.pop_weight*decay

    def dcor(self, x, y):

        a = torch.norm(x[:,None] - x, p = 2, dim = 2)
        b = torch.norm(y[:,None] - y, p = 2, dim = 2)

        A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
        B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask*torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask):

        p_score_int, n_score_int, kl_int = self.interest.pair_forward(user, item_p, item_n)
        p_score_pop, n_score_pop, kl_pop = self.popularity.pair_forward(user, item_p, item_n)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = self.interest.item_embedding(item_all)
        item_pop = self.popularity.item_embedding(item_all)
        user_all = torch.unique(user)
        user_int, _ = self.interest.encode(user_all, training=False)
        user_pop, _ = self.popularity.encode(user_all, training=False)
        discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)
        kl_loss = kl_int + kl_pop

        loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total + self.kl_weight*kl_loss - self.dis_pen*discrepency_loss

        return loss

    def score(self, user, item):

        return self.interest.score(user, item) + self.popularity.score(user, item)

    def score_interest(self, user, item):

        return self.interest.score(user, item)

    def score_popularity(self, user, item):

        return self.popularity.score(user, item)


class CausE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(CausE, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_control = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.items_treatment = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.criterion_factual = nn.BCEWithLogitsLoss()
        self.criterion_counterfactual = nn.MSELoss()

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items_control.data.uniform_(-stdv, stdv)
        self.items_treatment.data.uniform_(-stdv, stdv)

    def forward(self, user, item, label, mask):

        user_control = self.users[user[~mask]]
        item_control = self.items_control[item[~mask]]
        score_control = torch.sum(user_control * item_control, 2)
        label_control = label[~mask]
        control_loss = self.criterion_factual(score_control, label_control)

        control_distance = (torch.sigmoid(score_control) - label_control).abs().mean().item()

        user_treatment = self.users[user[mask]]
        item_treatment = self.items_treatment[item[mask]]
        score_treatment = torch.sum(user_treatment * item_treatment, 2)
        label_treatment = label[mask]
        treatment_loss = self.criterion_factual(score_treatment, label_treatment)

        treatment_distance = (torch.sigmoid(score_treatment) - label_treatment).abs().mean().item()

        item_all = torch.unique(item)
        item_control_factual = self.items_control[item_all]
        item_control_counterfactual = self.items_treatment[item_all]
        discrepency_loss = self.criterion_counterfactual(item_control_factual, item_control_counterfactual)

        return control_loss, treatment_loss, discrepency_loss, control_distance, treatment_distance

    def get_item_control_embeddings(self):

        return self.items_control.detach().cpu().numpy().astype('float32')

    def get_item_treatment_embeddings(self):

        return self.items_treatment.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        return self.users.detach().cpu().numpy().astype('float32')


class LGNCausE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout):

        super(LGNCausE, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.embeddings_control = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.embeddings_treatment = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers_control = nn.ModuleList()
        for _ in range(num_layers):
            self.layers_control.append(LGConv(embedding_size, embedding_size, 1))

        self.layers_treatment = nn.ModuleList()
        for _ in range(num_layers):
            self.layers_treatment.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        self.criterion_factual = nn.BCEWithLogitsLoss()
        self.criterion_counterfactual = nn.MSELoss()

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings_control.size(1))
        self.embeddings_control.data.uniform_(-stdv, stdv)
        self.embeddings_treatment.data.uniform_(-stdv, stdv)

    def forward(self, user, item, label, mask, graph_control, graph_treatment, training=True):

        features_control = [self.embeddings_control]
        h_control = self.embeddings_control
        for layer in self.layers_control:
            h_control = layer(graph_control, h_control)
            h_control = F.dropout(h_control, p=self.dropout, training=training)
            features_control.append(h_control)

        features_control = torch.stack(features_control, dim=2)
        features_control = torch.mean(features_control, dim=2)

        features_treatment = [self.embeddings_treatment]
        h_treatment = self.embeddings_treatment
        for layer in self.layers_treatment:
            h_treatment = layer(graph_treatment, h_treatment)
            h_treatment = F.dropout(h_treatment, p=self.dropout, training=training)
            features_treatment.append(h_treatment)

        features_treatment = torch.stack(features_treatment, dim=2)
        features_treatment = torch.mean(features_treatment, dim=2)

        item = item + self.n_user

        user_control = features_control[user[~mask]]
        item_control = features_control[item[~mask]]
        score_control = torch.sum(user_control * item_control, 2)
        label_control = label[~mask]
        control_loss = self.criterion_factual(score_control, label_control)

        control_distance = (torch.sigmoid(score_control) - label_control).abs().mean().item()

        user_treatment = features_treatment[user[mask]]
        item_treatment = features_treatment[item[mask]]
        score_treatment = torch.sum(user_treatment * item_treatment, 2)
        label_treatment = label[mask]
        treatment_loss = self.criterion_factual(score_treatment, label_treatment)

        treatment_distance = (torch.sigmoid(score_treatment) - label_treatment).abs().mean().item()

        user_control_factual = features_control[user]
        user_control_counterfactual = features_treatment[user]
        item_control_factual = features_control[item]
        item_control_counterfactual = features_treatment[item]
        discrepency_loss = self.criterion_counterfactual(user_control_factual, user_control_counterfactual) + self.criterion_counterfactual(item_control_factual, item_control_counterfactual)

        return control_loss, treatment_loss, discrepency_loss, control_distance, treatment_distance


    def get_control_embeddings(self, graph):

        features = [self.embeddings_control]
        h = self.embeddings_control
        for layer in self.layers_control:
            h = layer(graph, h)
            features.append(h)

        features = torch.stack(features, dim=2)
        features = torch.mean(features, dim=2)

        users = features[:self.n_user]
        items = features[self.n_user:]

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')


class LGConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None):
        super(LGConv, self).__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm

    def forward(self, graph, feat):
        if fn is None:
            raise ImportError('DGL is required for LGN models.')

        graph = graph.local_var()
        if self._cached_h is not None:
            feat = self._cached_h
        else:
            # compute normalization
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)
            # compute (D^-1 A^k D)^k X
            for _ in range(self._k):
                feat = feat * norm
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm

            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat

        return feat


class LGN(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout):

        super(LGN, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.embeddings = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings.size(1))
        self.embeddings.data.uniform_(-stdv, stdv)

    def pair_forward(self, user, item_p, item_n, graph, training=True):

        features = [self.embeddings]
        h = self.embeddings
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features.append(h)

        features = torch.stack(features, dim=2)
        features = torch.mean(features, dim=2)

        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        user = features[user]
        item_p = features[item_p]
        item_n = features[item_n]

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def get_embeddings(self, graph):

        features = [self.embeddings]
        h = self.embeddings
        for layer in self.layers:
            h = layer(graph, h)
            features.append(h)

        features = torch.stack(features, dim=2)
        features = torch.mean(features, dim=2)

        users = features[:self.n_user]
        items = features[self.n_user:]

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')


class NGCFConv(nn.Module):

    def __init__(self, embedding_size, dropout):

        super(NGCFConv, self).__init__()

        self.linear_gc = nn.Linear(embedding_size, embedding_size)
        self.linear_bi = nn.Linear(embedding_size, embedding_size)
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph, feat, training=True):

        side_feat = torch.sparse.mm(graph, feat)

        sum_embeddings = self.leaky_relu(self.linear_gc(side_feat))
        bi_embeddings = self.leaky_relu(self.linear_bi(feat * side_feat))
        out = sum_embeddings + bi_embeddings
        out = F.dropout(out, p=self.dropout, training=training)
        return F.normalize(out, p=2, dim=1)


class NGCF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout):

        super(NGCF, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.embeddings = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(NGCFConv(embedding_size, dropout))

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings.size(1))
        self.embeddings.data.uniform_(-stdv, stdv)

    def propagate(self, graph, training=True):

        features = [self.embeddings]
        h = self.embeddings
        for layer in self.layers:
            h = layer(graph, h, training=training)
            features.append(h)

        return torch.cat(features, dim=1)

    def pair_forward(self, user, item_p, item_n, graph, training=True):

        features = self.propagate(graph, training=training)

        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        user = features[user]
        item_p = features[item_p]
        item_n = features[item_n]

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def get_embeddings(self, graph):

        features = self.propagate(graph, training=False)

        users = features[:self.n_user]
        items = features[self.n_user:]

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')


class DICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, dis_loss, dis_pen, int_weight, pop_weight):

        super(DICE, self).__init__()

        self.users_int = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.users_pop = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_int = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.items_pop = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.int_weight = int_weight
        self.pop_weight = pop_weight

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.dis_pen = dis_pen

        self.init_params()

    def adapt(self, epoch, decay):

        self.int_weight = self.int_weight*decay
        self.pop_weight = self.pop_weight*decay

    def dcor(self, x, y):

        a = torch.norm(x[:,None] - x, p = 2, dim = 2)
        b = torch.norm(y[:,None] - y, p = 2, dim = 2)

        A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
        B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean() 

        n = x.size(0)

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):

        stdv = 1. / math.sqrt(self.users_int.size(1))
        self.users_int.data.uniform_(-stdv, stdv)
        self.users_pop.data.uniform_(-stdv, stdv)
        self.items_int.data.uniform_(-stdv, stdv)
        self.items_pop.data.uniform_(-stdv, stdv)

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask*torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask):

        users_int = self.users_int[user]
        users_pop = self.users_pop[user]
        items_p_int = self.items_int[item_p]
        items_p_pop = self.items_pop[item_p]
        items_n_int = self.items_int[item_n]
        items_n_pop = self.items_pop[item_n]

        p_score_int = torch.sum(users_int*items_p_int, 2)
        n_score_int = torch.sum(users_int*items_n_int, 2)

        p_score_pop = torch.sum(users_pop*items_p_pop, 2)
        n_score_pop = torch.sum(users_pop*items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = self.items_int[item_all]
        item_pop = self.items_pop[item_all]
        user_all = torch.unique(user)
        user_int = self.users_int[user_all]
        user_pop = self.users_pop[user_all]
        discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

        loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total - self.dis_pen*discrepency_loss

        return loss

    def get_item_embeddings(self):

        item_embeddings = torch.cat((self.items_int, self.items_pop), 1)
        #item_embeddings = self.items_pop
        return item_embeddings.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        user_embeddings = torch.cat((self.users_int, self.users_pop), 1)
        #user_embeddings = self.users_pop
        return user_embeddings.detach().cpu().numpy().astype('float32')


class IDICE(DICE):

    def __init__(
        self,
        num_users,
        num_items,
        embedding_size,
        dis_loss,
        dis_pen,
        int_weight,
        pop_weight,
        social_weight,
        social_reg_weight,
    ):

        super(IDICE, self).__init__(num_users, num_items, embedding_size, dis_loss, dis_pen, int_weight, pop_weight)

        self.users_social = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_social = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.social_weight = social_weight
        self.social_reg_weight = social_reg_weight

        stdv = 1. / math.sqrt(embedding_size)
        self.users_social.data.uniform_(-stdv, stdv)
        self.items_social.data.uniform_(-stdv, stdv)

    def adapt(self, epoch, decay):

        super(IDICE, self).adapt(epoch, decay)
        self.social_weight = self.social_weight*decay

    def forward(self, user, item_p, item_n, mask, social_edges=None):

        users_int = self.users_int[user]
        users_pop = self.users_pop[user]
        users_social = self.users_social[user]
        items_p_int = self.items_int[item_p]
        items_p_pop = self.items_pop[item_p]
        items_p_social = self.items_social[item_p]
        items_n_int = self.items_int[item_n]
        items_n_pop = self.items_pop[item_n]
        items_n_social = self.items_social[item_n]

        p_score_int = torch.sum(users_int*items_p_int, 2)
        n_score_int = torch.sum(users_int*items_n_int, 2)

        p_score_pop = torch.sum(users_pop*items_p_pop, 2)
        n_score_pop = torch.sum(users_pop*items_n_pop, 2)

        p_score_social = torch.sum(users_social*items_p_social, 2)
        n_score_social = torch.sum(users_social*items_n_social, 2)

        p_score_total = p_score_int + p_score_pop + self.social_weight*p_score_social
        n_score_total = n_score_int + n_score_pop + self.social_weight*n_score_social

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
        loss_social = self.bpr_loss(p_score_social, n_score_social)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = self.items_int[item_all]
        item_pop = self.items_pop[item_all]
        item_social = self.items_social[item_all]
        user_all = torch.unique(user)
        user_int = self.users_int[user_all]
        user_pop = self.users_pop[user_all]
        user_social = self.users_social[user_all]

        discrepency_loss = (
            self.criterion_discrepancy(item_int, item_pop)
            + self.criterion_discrepancy(user_int, user_pop)
            + self.criterion_discrepancy(item_int, item_social)
            + self.criterion_discrepancy(user_int, user_social)
        )

        social_reg = user_social.new_tensor(0.0)
        if social_edges is not None and social_edges.numel() > 0:
            src = social_edges[:, 0]
            dst = social_edges[:, 1]
            social_reg = torch.mean(torch.sum((self.users_social[src] - self.users_social[dst]) ** 2, dim=1))

        loss = (
            self.int_weight*loss_int
            + self.pop_weight*loss_pop
            + self.social_weight*loss_social
            + loss_total
            - self.dis_pen*discrepency_loss
            + self.social_reg_weight*social_reg
        )

        return loss

    def get_item_embeddings(self):

        social_scale = math.sqrt(max(float(self.social_weight), 0.0))
        item_embeddings = torch.cat((self.items_int, self.items_pop, social_scale*self.items_social), 1)
        return item_embeddings.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        social_scale = math.sqrt(max(float(self.social_weight), 0.0))
        user_embeddings = torch.cat((self.users_int, self.users_pop, social_scale*self.users_social), 1)
        return user_embeddings.detach().cpu().numpy().astype('float32')


class LGNDICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout, dis_loss, dis_pen, int_weight, pop_weight):

        super(LGNDICE, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.int_weight = int_weight
        self.pop_weight = pop_weight

        self.embeddings_int = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.embeddings_pop = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.dis_pen = dis_pen

        self.init_params()

    def dcor(self, x, y):

        a = torch.norm(x[:,None] - x, p = 2, dim = 2)
        b = torch.norm(y[:,None] - y, p = 2, dim = 2)

        A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
        B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean() 

        n = x.size(0)

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings_int.size(1))
        self.embeddings_int.data.uniform_(-stdv, stdv)
        self.embeddings_pop.data.uniform_(-stdv, stdv)

    def adapt(self, epoch, decay):

        self.int_weight = self.int_weight*decay
        self.pop_weight = self.pop_weight*decay

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask*torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask, graph, training=True):

        features_int = [self.embeddings_int]
        h = self.embeddings_int
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_int.append(h)

        features_int = torch.stack(features_int, dim=2)
        features_int = torch.mean(features_int, dim=2)

        features_pop = [self.embeddings_pop]
        h = self.embeddings_pop
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_pop.append(h)

        features_pop = torch.stack(features_pop, dim=2)
        features_pop = torch.mean(features_pop, dim=2)

        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        users_int = features_int[user]
        users_pop = features_pop[user]
        items_p_int = features_int[item_p]
        items_p_pop = features_pop[item_p]
        items_n_int = features_int[item_n]
        items_n_pop = features_pop[item_n]

        p_score_int = torch.sum(users_int*items_p_int, 2)
        n_score_int = torch.sum(users_int*items_n_int, 2)

        p_score_pop = torch.sum(users_pop*items_p_pop, 2)
        n_score_pop = torch.sum(users_pop*items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = features_int[item_all]
        item_pop = features_pop[item_all]
        user_all = torch.unique(user)
        user_int = features_int[user_all]
        user_pop = features_pop[user_all]
        discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

        loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total - self.dis_pen*discrepency_loss

        return loss

    def get_embeddings(self, graph):

        features_int = [self.embeddings_int]
        h = self.embeddings_int
        for layer in self.layers:
            h = layer(graph, h)
            features_int.append(h)

        features_int = torch.stack(features_int, dim=2)
        features_int = torch.mean(features_int, dim=2)

        users_int = features_int[:self.n_user]
        items_int = features_int[self.n_user:]

        features_pop = [self.embeddings_pop]
        h = self.embeddings_pop
        for layer in self.layers:
            h = layer(graph, h)
            features_pop.append(h)

        features_pop = torch.stack(features_pop, dim=2)
        features_pop = torch.mean(features_pop, dim=2)
        users_pop = features_pop[:self.n_user]
        items_pop = features_pop[self.n_user:]

        items = torch.cat((items_int, items_pop), 1)
        users = torch.cat((users_int, users_pop), 1)

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')


class NGCFDICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout, dis_loss, dis_pen, int_weight, pop_weight):

        super(NGCFDICE, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.int_weight = int_weight
        self.pop_weight = pop_weight

        self.embeddings_int = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.embeddings_pop = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers_int = nn.ModuleList()
        self.layers_pop = nn.ModuleList()
        for _ in range(num_layers):
            self.layers_int.append(NGCFConv(embedding_size, dropout))
            self.layers_pop.append(NGCFConv(embedding_size, dropout))

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.dis_pen = dis_pen

        self.init_params()

    def dcor(self, x, y):

        a = torch.norm(x[:,None] - x, p = 2, dim = 2)
        b = torch.norm(y[:,None] - y, p = 2, dim = 2)

        A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
        B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings_int.size(1))
        self.embeddings_int.data.uniform_(-stdv, stdv)
        self.embeddings_pop.data.uniform_(-stdv, stdv)

    def adapt(self, epoch, decay):

        self.int_weight = self.int_weight*decay
        self.pop_weight = self.pop_weight*decay

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask*torch.log(torch.sigmoid(p_score - n_score)))

    def propagate(self, graph, embeddings, layers, training=True):

        features = [embeddings]
        h = embeddings
        for layer in layers:
            h = layer(graph, h, training=training)
            features.append(h)

        return torch.cat(features, dim=1)

    def forward(self, user, item_p, item_n, mask, graph, training=True):

        features_int = self.propagate(graph, self.embeddings_int, self.layers_int, training=training)
        features_pop = self.propagate(graph, self.embeddings_pop, self.layers_pop, training=training)

        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        users_int = features_int[user]
        users_pop = features_pop[user]
        items_p_int = features_int[item_p]
        items_p_pop = features_pop[item_p]
        items_n_int = features_int[item_n]
        items_n_pop = features_pop[item_n]

        p_score_int = torch.sum(users_int*items_p_int, 2)
        n_score_int = torch.sum(users_int*items_n_int, 2)

        p_score_pop = torch.sum(users_pop*items_p_pop, 2)
        n_score_pop = torch.sum(users_pop*items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = features_int[item_all]
        item_pop = features_pop[item_all]
        user_all = torch.unique(user)
        user_int = features_int[user_all]
        user_pop = features_pop[user_all]
        discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

        loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total - self.dis_pen*discrepency_loss

        return loss

    def get_embeddings(self, graph):

        features_int = self.propagate(graph, self.embeddings_int, self.layers_int, training=False)

        users_int = features_int[:self.n_user]
        items_int = features_int[self.n_user:]

        features_pop = self.propagate(graph, self.embeddings_pop, self.layers_pop, training=False)
        users_pop = features_pop[:self.n_user]
        items_pop = features_pop[self.n_user:]

        items = torch.cat((items_int, items_pop), 1)
        users = torch.cat((users_int, users_pop), 1)

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')

