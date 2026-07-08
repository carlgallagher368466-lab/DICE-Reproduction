#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

import data
import model
import utils
import candidate_generator as cg
import config.const as const_util

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
try:
    import dgl
except ImportError:
    dgl = None

import data_utils.loader as LOADER


class Recommender(object):

    def __init__(self, flags_obj, workspace, dm):

        self.dm = dm
        self.model_name = flags_obj.model
        self.flags_obj = flags_obj
        self.set_device()
        self.set_model()
        self.workspace = workspace

    def set_device(self):

        self.device  = utils.ContextManager.set_device(self.flags_obj)

    def set_model(self):

        raise NotImplementedError

    def transfer_model(self):

        self.model = self.model.to(self.device)

    def save_ckpt(self, epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        torch.save(self.model.state_dict(), model_path)

    def cleanup_ckpt(self, keep_epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if keep_epoch is None or keep_epoch < 0 or not os.path.exists(ckpt_path):
            return

        keep_name = 'epoch_' + str(keep_epoch) + '.pth'
        for filename in os.listdir(ckpt_path):
            if not filename.startswith('epoch_') or not filename.endswith('.pth'):
                continue
            if filename == keep_name:
                continue
            os.remove(os.path.join(ckpt_path, filename))

    def load_ckpt(self, epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def get_dataloader(self):

        raise NotImplementedError

    def get_pair_dataloader(self):

        raise NotImplementedError

    def get_point_dataloader(self):

        raise NotImplementedError

    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.flags_obj.lr, weight_decay=self.flags_obj.weight_decay, betas=(0.5, 0.99), amsgrad=True)

    def inference(self, sample):

        raise NotImplementedError

    def make_cg(self):

        raise NotImplementedError

    def cg(self, users, topk):

        raise NotImplementedError


class MFRecommender(Recommender):

    def __init__(self, flags_obj, workspace, dm):

        super(MFRecommender, self).__init__(flags_obj, workspace, dm)
        self.dm.get_skew_dataset()

    def set_model(self):

        self.model = model.MF(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size)

    def get_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_blend_pair_dataloader(self.flags_obj, self.dm)

    def pair_inference(self, sample):

        user, item_p, item_n = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)

        p_score, n_score = self.model.pair_forward(user, item_p, item_n)

        return p_score, n_score

    def make_cg(self):

        self.item_embeddings = self.model.get_item_embeddings()
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)

        self.user_embeddings = self.model.get_user_embeddings()

    def cg(self, users, topk):

        return self.generator.generate(self.user_embeddings[users], topk)


class IPSRecommender(MFRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(IPSRecommender, self).__init__(flags_obj, workspace, dm)
        self.dm.get_skew_dataset()

    def get_ips_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_ips_blend_pair_dataloader(self.flags_obj, self.dm)

    def pair_inference(self, sample):

        user, item_p, item_n, weight = sample
        sample_wrap = (user, item_p, item_n)
        p_score, n_score = super(IPSRecommender, self).pair_inference(sample_wrap)
        weight = weight.to(self.device)

        return p_score, n_score, weight


class NeuMFRecommender(MFRecommender):

    def set_model(self):

        self.model = model.NeuMF(
            self.dm.n_user,
            self.dm.n_item,
            self.flags_obj.embedding_size,
            self.flags_obj.neumf_layers,
            self.flags_obj.neumf_dropout,
        )

    def make_cg(self):

        self.generator = cg.TorchScoringTopKGenerator(
            self.model,
            self.dm.n_item,
            self.device,
            self.flags_obj.neumf_item_chunk_size,
        )

    def cg(self, users, topk):

        return self.generator.generate(users, topk)


class NeuMFIPSRecommender(NeuMFRecommender):

    def get_ips_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_ips_blend_pair_dataloader(self.flags_obj, self.dm)

    def pair_inference(self, sample):

        user, item_p, item_n, weight = sample
        sample_wrap = (user, item_p, item_n)
        p_score, n_score = super(NeuMFIPSRecommender, self).pair_inference(sample_wrap)
        weight = weight.to(self.device)

        return p_score, n_score, weight


class NeuMFDICERecommender(MFRecommender):

    def set_model(self):

        self.model = model.NeuMFDICE(
            self.dm.n_user,
            self.dm.n_item,
            self.flags_obj.embedding_size,
            self.flags_obj.neumf_layers,
            self.flags_obj.neumf_dropout,
            self.flags_obj.dis_loss,
            self.flags_obj.dis_pen,
            self.flags_obj.int_weight,
            self.flags_obj.pop_weight,
        )

    def make_cg(self):

        self.generator = cg.TorchScoringTopKGenerator(
            self.model,
            self.dm.n_item,
            self.device,
            self.flags_obj.neumf_item_chunk_size,
        )

    def cg(self, users, topk):

        return self.generator.generate(users, topk)

    def get_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_DICE_dataloader(self.flags_obj, self.dm)

    def get_loss(self, sample):

        user, item_p, item_n, mask = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)
        mask = mask.to(self.device)

        loss = self.model(user, item_p, item_n, mask)

        return loss

    def adapt(self, epoch, decay):

        self.model.adapt(epoch, decay)


class VAERecommender(MFRecommender):

    def set_model(self):

        self.model = model.VAE(
            self.dm.n_user,
            self.dm.n_item,
            self.flags_obj.embedding_size,
            self.flags_obj.vae_latent_size,
            self.flags_obj.vae_hidden_size,
            self.flags_obj.vae_dropout,
        )

    def pair_inference(self, sample):

        user, item_p, item_n = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)

        p_score, n_score, kl_loss = self.model.pair_forward(user, item_p, item_n)

        return p_score, n_score, kl_loss

    def make_cg(self):

        self.generator = cg.TorchScoringTopKGenerator(
            self.model,
            self.dm.n_item,
            self.device,
            self.flags_obj.neumf_item_chunk_size,
        )

    def cg(self, users, topk):

        return self.generator.generate(users, topk)


class VAEIPSRecommender(VAERecommender):

    def get_ips_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_ips_blend_pair_dataloader(self.flags_obj, self.dm)

    def pair_inference(self, sample):

        user, item_p, item_n, weight = sample
        sample_wrap = (user, item_p, item_n)
        p_score, n_score, kl_loss = super(VAEIPSRecommender, self).pair_inference(sample_wrap)
        weight = weight.to(self.device)

        return p_score, n_score, weight, kl_loss


class VAEDICERecommender(MFRecommender):

    def set_model(self):

        self.model = model.VAEDICE(
            self.dm.n_user,
            self.dm.n_item,
            self.flags_obj.embedding_size,
            self.flags_obj.vae_latent_size,
            self.flags_obj.vae_hidden_size,
            self.flags_obj.vae_dropout,
            self.flags_obj.dis_loss,
            self.flags_obj.dis_pen,
            self.flags_obj.int_weight,
            self.flags_obj.pop_weight,
            self.flags_obj.vae_kl_weight,
        )

    def make_cg(self):

        self.generator = cg.TorchScoringTopKGenerator(
            self.model,
            self.dm.n_item,
            self.device,
            self.flags_obj.neumf_item_chunk_size,
        )

    def cg(self, users, topk):

        return self.generator.generate(users, topk)

    def get_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_DICE_dataloader(self.flags_obj, self.dm)

    def get_loss(self, sample):

        user, item_p, item_n, mask = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)
        mask = mask.to(self.device)

        loss = self.model(user, item_p, item_n, mask)

        return loss

    def adapt(self, epoch, decay):

        self.model.adapt(epoch, decay)


class CausERecommender(Recommender):

    def __init__(self, flags_obj, workspace, dm):

        super(CausERecommender, self).__init__(flags_obj, workspace, dm)
        self.dm.get_skew_dataset()

    def set_model(self):

        self.model = model.CausE(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size)

    def get_point_dataloader(self):

        return data.FactorizationDataProcessor.get_CausE_dataloader(self.flags_obj, self.dm)

    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.flags_obj.lr, weight_decay=self.flags_obj.weight_decay, betas=(0.5, 0.99), amsgrad=True)

    def get_loss(self, sample):

        user, item, label, mask = sample

        user = user.to(self.device)
        item = item.to(self.device)
        label = label.to(self.device)
        mask = torch.squeeze(mask)
        mask = mask.to(self.device)

        control_loss, treatment_loss, discrepency_loss, control_distance, treatment_distance = self.model(user, item, label, mask)
        loss = control_loss + treatment_loss + self.flags_obj.dis_pen*discrepency_loss

        return loss, control_loss, treatment_loss, discrepency_loss, control_distance, treatment_distance

    def make_cg(self):

        self.item_embeddings = self.model.get_item_control_embeddings()
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)

        self.user_embeddings = self.model.get_user_embeddings()

    def cg(self, users, topk):

        return self.generator.generate(self.user_embeddings[users], topk)


class LGNCausERecommender(CausERecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(LGNCausERecommender, self).__init__(flags_obj, workspace, dm)
        self.init_graph(flags_obj)

    def init_graph(self, flags_obj):
        if dgl is None:
            raise ImportError('DGL is required for LGN models.')

        coo_loader = LOADER.CooLoader(flags_obj)

        self.coo_adj_graph = coo_loader.load(const_util.train_coo_adj_graph)

        self.graph_control = dgl.DGLGraph()

        num_nodes = self.coo_adj_graph.shape[0]
        self.graph_control.add_nodes(num_nodes)
        self.graph_control.ndata['feature'] = torch.arange(num_nodes)

        self.graph_control.add_edges(self.coo_adj_graph.row, self.coo_adj_graph.col)
        self.graph_control.add_edges(self.graph_control.nodes(), self.graph_control.nodes())

        self.graph_control.readonly()

        self.skew_coo_adj_graph = coo_loader.load(const_util.train_skew_coo_adj_graph)

        self.graph_treatment = dgl.DGLGraph()

        num_nodes = self.skew_coo_adj_graph.shape[0]
        self.graph_treatment.add_nodes(num_nodes)
        self.graph_treatment.ndata['feature'] = torch.arange(num_nodes)

        self.graph_treatment.add_edges(self.skew_coo_adj_graph.row, self.skew_coo_adj_graph.col)
        self.graph_treatment.add_edges(self.graph_treatment.nodes(), self.graph_treatment.nodes())

        self.graph_treatment.readonly()

    def set_model(self):

        self.model = model.LGNCausE(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size, self.flags_obj.num_layers, self.flags_obj.dropout)

    def get_loss(self, sample):

        user, item, label, mask = sample

        user = user.to(self.device)
        item = item.to(self.device)
        label = label.to(self.device)
        mask = torch.squeeze(mask)
        mask = mask.to(self.device)

        control_loss, treatment_loss, discrepency_loss, control_distance, treatment_distance = self.model(user, item, label, mask, self.graph_control, self.graph_treatment)
        loss = control_loss + treatment_loss + self.flags_obj.dis_pen*discrepency_loss

        return loss, control_loss, treatment_loss, discrepency_loss, control_distance, treatment_distance

    def make_cg(self):

        self.item_embeddings, self.user_embeddings = self.model.get_control_embeddings(self.graph_control)
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)


class LGNRecommender(MFRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(LGNRecommender, self).__init__(flags_obj, workspace, dm)
        self.init_graph(flags_obj)

    def init_graph(self, flags_obj):
        if dgl is None:
            raise ImportError('DGL is required for LGN models.')

        coo_loader = LOADER.CooLoader(flags_obj)
        self.coo_adj_graph = coo_loader.load(const_util.train_blend_coo_adj_graph)

        self.graph = dgl.DGLGraph()

        num_nodes = self.coo_adj_graph.shape[0]
        self.graph.add_nodes(num_nodes)
        self.graph.ndata['feature'] = torch.arange(num_nodes)

        self.graph.add_edges(self.coo_adj_graph.row, self.coo_adj_graph.col)
        self.graph.add_edges(self.graph.nodes(), self.graph.nodes())

        self.graph.readonly()

    def set_model(self):

        self.model = model.LGN(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size, self.flags_obj.num_layers, self.flags_obj.dropout)

    def pair_inference(self, sample):

        user, item_p, item_n = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)

        p_score, n_score = self.model.pair_forward(user, item_p, item_n, self.graph)

        return p_score, n_score

    def make_cg(self):

        self.item_embeddings, self.user_embeddings = self.model.get_embeddings(self.graph)
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)


class LGNIPSRecommender(LGNRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(LGNIPSRecommender, self).__init__(flags_obj, workspace, dm)

    def get_ips_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_ips_blend_pair_dataloader(self.flags_obj, self.dm)

    def pair_inference(self, sample):

        user, item_p, item_n, weight = sample
        sample_wrap = (user, item_p, item_n)
        p_score, n_score = super(LGNIPSRecommender, self).pair_inference(sample_wrap)
        weight = weight.to(self.device)

        return p_score, n_score, weight


class NGCFRecommender(LGNRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(LGNRecommender, self).__init__(flags_obj, workspace, dm)
        self.dm.get_skew_dataset()
        self.init_graph(flags_obj)

    def set_model(self):

        self.model = model.NGCF(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size, self.flags_obj.num_layers, self.flags_obj.dropout)

    def init_graph(self, flags_obj):

        coo_loader = LOADER.CooLoader(flags_obj)
        coo_adj_graph = coo_loader.load(const_util.train_blend_coo_adj_graph).tocoo()
        self.graph = self.make_sparse_norm_adj(coo_adj_graph)

    def make_sparse_norm_adj(self, coo_adj_graph):

        coo_adj_graph = (coo_adj_graph + sp.eye(coo_adj_graph.shape[0], dtype=coo_adj_graph.dtype, format='coo')).tocoo()
        rows = torch.from_numpy(coo_adj_graph.row.astype(np.int64))
        cols = torch.from_numpy(coo_adj_graph.col.astype(np.int64))
        values = torch.from_numpy(coo_adj_graph.data.astype(np.float32))
        degree = torch.zeros(coo_adj_graph.shape[0], dtype=torch.float32)
        degree.scatter_add_(0, rows, values)
        degree = degree.clamp(min=1)
        norm_values = values * torch.pow(degree[rows], -0.5) * torch.pow(degree[cols], -0.5)
        indices = torch.stack([rows, cols], dim=0)
        graph = torch.sparse_coo_tensor(indices, norm_values, coo_adj_graph.shape).coalesce()
        return graph.to(self.device)


class NGCFIPSRecommender(NGCFRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(NGCFIPSRecommender, self).__init__(flags_obj, workspace, dm)

    def get_ips_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_ips_blend_pair_dataloader(self.flags_obj, self.dm)

    def pair_inference(self, sample):

        user, item_p, item_n, weight = sample
        sample_wrap = (user, item_p, item_n)
        p_score, n_score = super(NGCFIPSRecommender, self).pair_inference(sample_wrap)
        weight = weight.to(self.device)

        return p_score, n_score, weight


class DICERecommender(MFRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(DICERecommender, self).__init__(flags_obj, workspace, dm)

    def set_model(self):

        self.model = model.DICE(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size, self.flags_obj.dis_loss, self.flags_obj.dis_pen, self.flags_obj.int_weight, self.flags_obj.pop_weight)

    def get_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_DICE_dataloader(self.flags_obj, self.dm)

    def get_loss(self, sample):

        user, item_p, item_n, mask = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)
        mask = mask.to(self.device)

        loss = self.model(user, item_p, item_n, mask)

        return loss

    def make_cg(self):

        self.item_embeddings = self.model.get_item_embeddings()
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)

        self.user_embeddings = self.model.get_user_embeddings()

    def cg(self, users, topk):

        return self.generator.generate(self.user_embeddings[users], topk)

    def adapt(self, epoch, decay):

        self.model.adapt(epoch, decay)


class IDICERecommender(DICERecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(IDICERecommender, self).__init__(flags_obj, workspace, dm)
        self.social_edges = self.load_social_edges(flags_obj)

    def set_model(self):

        self.model = model.IDICE(
            self.dm.n_user,
            self.dm.n_item,
            self.flags_obj.embedding_size,
            self.flags_obj.dis_loss,
            self.flags_obj.dis_pen,
            self.flags_obj.int_weight,
            self.flags_obj.pop_weight,
            self.flags_obj.social_weight,
            self.flags_obj.social_reg_weight,
        )

    def load_social_edges(self, flags_obj):

        social_path = os.path.join(flags_obj.load_path, const_util.social_edges)
        if not os.path.exists(social_path):
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        social_edges = pd.read_csv(social_path)
        if social_edges.empty:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        social_edges = social_edges[['src', 'dst']].to_numpy(dtype=np.int64)
        return torch.LongTensor(social_edges).to(self.device)

    def get_loss(self, sample):

        user, item_p, item_n, mask = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)
        mask = mask.to(self.device)

        loss = self.model(user, item_p, item_n, mask, self.social_edges)

        return loss


class LGNDICERecommender(DICERecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(LGNDICERecommender, self).__init__(flags_obj, workspace, dm)
        self.init_graph(flags_obj)

    def init_graph(self, flags_obj):
        if dgl is None:
            raise ImportError('DGL is required for LGN models.')

        coo_loader = LOADER.CooLoader(flags_obj)
        self.coo_adj_graph = coo_loader.load(const_util.train_blend_coo_adj_graph)

        self.graph = dgl.DGLGraph()

        num_nodes = self.coo_adj_graph.shape[0]
        self.graph.add_nodes(num_nodes)
        self.graph.ndata['feature'] = torch.arange(num_nodes)

        self.graph.add_edges(self.coo_adj_graph.row, self.coo_adj_graph.col)
        self.graph.add_edges(self.graph.nodes(), self.graph.nodes())

        self.graph.readonly()

    def set_model(self):

        self.model = model.LGNDICE(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size, self.flags_obj.num_layers, self.flags_obj.dropout, self.flags_obj.dis_loss, self.flags_obj.dis_pen, self.flags_obj.int_weight, self.flags_obj.pop_weight)

    def get_loss(self, sample):

        user, item_p, item_n, mask = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)
        mask = mask.to(self.device)

        loss = self.model(user, item_p, item_n, mask, self.graph)

        return loss

    def make_cg(self):

        self.item_embeddings, self.user_embeddings = self.model.get_embeddings(self.graph)
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)

    def adapt(self, epoch, decay):

        self.model.adapt(epoch, decay)


class NGCFDICERecommender(LGNDICERecommender):

    def __init__(self, flags_obj, workspace, dm):

        DICERecommender.__init__(self, flags_obj, workspace, dm)
        self.init_graph(flags_obj)

    def set_model(self):

        self.model = model.NGCFDICE(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size, self.flags_obj.num_layers, self.flags_obj.dropout, self.flags_obj.dis_loss, self.flags_obj.dis_pen, self.flags_obj.int_weight, self.flags_obj.pop_weight)

    def init_graph(self, flags_obj):

        coo_loader = LOADER.CooLoader(flags_obj)
        coo_adj_graph = coo_loader.load(const_util.train_blend_coo_adj_graph).tocoo()
        self.graph = NGCFRecommender.make_sparse_norm_adj(self, coo_adj_graph)


class PopularityRecommender(Recommender):

    def __init__(self, flags_obj, workspace, dm):

        super(PopularityRecommender, self).__init__(flags_obj, workspace, dm)

    def set_model(self):

        pass

    def transfer_model(self):

        pass

    def load_ckpt(self, epoch):

        pass

    def make_cg(self):

        popularity = self.dm.get_popularity()
        self.generator = cg.PopularityGenerator(self.flags_obj, popularity, 500)

    def cg(self, users, topk):

        return self.generator.generate(users, topk) 
