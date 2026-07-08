#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags
from absl import logging

import sys
sys.path.append('/home/zhengyu/workspace/DICE')

import utils
from trainer import Trainer
from tester import Tester

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'MF-debug', 'Experiment name.')
flags.DEFINE_enum('model', 'DICE', ['MF', 'DICE', 'IDICE', 'IPS', 'CausE', 'LGN', 'LGNDICE', 'LGNIPS', 'LGNCausE', 'NeuMF', 'NeuMFIPS', 'NeuMFDICE', 'NGCF', 'NGCFIPS', 'NGCFDICE', 'VAE', 'VAEIPS', 'VAEDICE'], 'Model name.')
flags.DEFINE_integer('num_layers', 2, 'The number of layers for LGN.')
flags.DEFINE_float('dropout', 0.2, 'Dropout ratio for LGN.')
flags.DEFINE_integer('margin', 40, 'Margin for negative sampling.')
flags.DEFINE_integer('pool', 40, 'Pool for negative sampling.')
flags.DEFINE_bool('adaptive', False, 'Adapt hyper-parameters or not.')
flags.DEFINE_float('margin_decay', 0.9, 'Decay of margin and pool.')
flags.DEFINE_float('loss_decay', 0.9, 'Decay of loss.')
flags.DEFINE_enum('weighting_mode', 'nc', ['n', 'c', 'nc', 'x'], 'Mode of IPS technique.')
flags.DEFINE_float('weighting_smoothness', 1.0, 'IPS weighting smoothness.')
flags.DEFINE_bool('use_gpu', True, 'Use GPU or not.')
flags.DEFINE_integer('gpu_id', 6, 'GPU ID.')
flags.DEFINE_bool('cg_use_gpu', True, 'Use GPU or not for candidate generation.')
flags.DEFINE_integer('cg_gpu_id', 1, 'GPU ID for candidate generation.')
flags.DEFINE_enum('dataset', 'ml10m', ['ml10m', 'nf', 'ml10m_longtail_test', 'ml10m_head_test', 'ml10m_uniform_test', 'ciao', 'epinions'], 'Dataset.')
flags.DEFINE_integer('embedding_size', 64, 'Embedding size for embedding based models.')
flags.DEFINE_multi_integer('neumf_layers', [128, 64, 32], 'MLP hidden sizes for NeuMF.')
flags.DEFINE_float('neumf_dropout', 0.0, 'Dropout ratio for NeuMF MLP.')
flags.DEFINE_integer('neumf_item_chunk_size', 2048, 'Item chunk size for NeuMF full scoring candidate generation.')
flags.DEFINE_integer('vae_latent_size', 64, 'Latent size for VAE.')
flags.DEFINE_integer('vae_hidden_size', 128, 'Hidden size for VAE encoder.')
flags.DEFINE_float('vae_dropout', 0.0, 'Dropout ratio for VAE encoder.')
flags.DEFINE_float('vae_kl_weight', 0.001, 'KL loss weight for VAE-DICE.')
flags.DEFINE_integer('epochs', 500, 'Max epochs for training.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('min_lr', 0.0001, 'Minimum learning rate.')
flags.DEFINE_float('weight_decay', 5e-8, 'Weight decay.')
flags.DEFINE_integer('batch_size', 128, 'Batch Size.')
flags.DEFINE_enum('dis_loss', 'dcor', ['L1', 'L2', 'dcor'], 'Discrepency loss function.')
flags.DEFINE_float('dis_pen', 0.01, 'Discrepency penalty.')
flags.DEFINE_float('int_weight', 0.1, 'Weight for interest term.')
flags.DEFINE_float('pop_weight', 0.1, 'Weight for popularity term.')
flags.DEFINE_float('social_weight', 0.1, 'Weight for social influence term.')
flags.DEFINE_float('social_reg_weight', 0.01, 'Weight for social trust regularization.')
flags.DEFINE_integer('neg_sample_rate', 4, 'Negative Sampling Ratio.')
flags.DEFINE_bool('shuffle', True, 'Shuffle the training set or not.')
flags.DEFINE_multi_string('metrics', ['recall', 'hit_ratio', 'ndcg'], 'Metrics.')
flags.DEFINE_multi_string('val_metrics', ['recall', 'hit_ratio', 'ndcg'], 'Metrics.')
flags.DEFINE_string('watch_metric', 'recall', 'Which metric to decide learning rate reduction.')
flags.DEFINE_integer('patience', 5, 'Patience for reducing learning rate.')
flags.DEFINE_integer('es_patience', 3, 'Patience for early stop.')
flags.DEFINE_integer('num_val_users', 1000000, 'Number of users for validation.')
flags.DEFINE_integer('num_test_users', 1000000, 'Number of users for test.')
flags.DEFINE_enum('test_model', 'best', ['best', 'last'], 'Which model to test.')
flags.DEFINE_multi_integer('topk', [20, 50], 'Topk for testing recommendation performance.')
flags.DEFINE_integer('num_workers', 8, 'Number of processes for training and testing.')
flags.DEFINE_string('load_path', '', 'Load path.')
flags.DEFINE_string('workspace', './', 'Path to load ckpt.')
flags.DEFINE_string('output', '/home/zhengyu/workspace/DICE/output/', 'Directory to save model/log/metrics.')
flags.DEFINE_string('dump_recommendations_dir', '', 'Directory to save filtered test recommendations as CSV.')
flags.DEFINE_integer('port', 33336, 'Port to show visualization results.')
flags.DEFINE_bool('use_visdom', False, 'Use visdom visualization or not.')


def main(argv):

    flags_obj = FLAGS
    cm = utils.ContextManager(flags_obj)
    vm = utils.VizManager(flags_obj)
    dm = utils.DatasetManager(flags_obj)
    dm.get_dataset_info()

    cm.set_default_ui()
    cm.logging_flags(flags_obj)
    vm.show_basic_info(flags_obj)
    trainer = utils.ContextManager.set_trainer(flags_obj, cm, vm, dm)
    trainer.train()

    trainer.test()


if __name__ == "__main__":

    app.run(main)

