import collections
import os
import sys

import torch
from torch.nn import functional as F

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sys_config import EXP_DIR

def print_stats(labels_list, name):
    """

    :param stats_list: format: [(class_i, %), (class_i+1, %), ...]
    :return:
    """
    c = collections.Counter(labels_list)
    stats_list = [(i, c[i] / len(labels_list) * 100.0) for i in c]

    str = "{} set stats: ".format(name)
    for i in range(0, len(stats_list)):
        str += "class {}: {}% ".format(stats_list[i][0], round(stats_list[i][1]),3)

    print(str)
    return

def number_h(num):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, 'Yi')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('Created {}'.format(dir_path))
        return

def create_exp_dirs(args):
    model_type = args.model_type if args.model_type != 'allenai/scibert' else 'bert'

    exp_name = 'al_{}_{}_{}'.format(args.dataset_name,
                                          model_type,
                                          args.acquisition)

    if args.indicator is not None: exp_name += '_{}'.format(args.indicator)

    al_iterations_dir = EXP_DIR
    create_dir(al_iterations_dir)
    # /experiments/{dataset}_{model}_{acquisition}
    al_config_dir = os.path.join(al_iterations_dir, exp_name)
    create_dir(al_config_dir)
    # /experiments/al_iterations/{dataset}_{model}_{acquisition}/{variant}_{seed}
    var_seed = '{}'.format(args.seed)
    if args.indicator is not None: var_seed += '_{}'.format(args.indicator)
    if args.mc_samples is None and args.acquisition in ['entropy', 'least_conf']:
        var_seed += '_vanilla'
    # if args.debug: var_seed += '_debug'
    if args.reverse: var_seed += '_reverse'
    if args.mean_embs: var_seed += '_inputs'
    if args.mean_out: var_seed += '_outputs'
    if args.cls: var_seed += '_cls'
    if args.ce: var_seed += '_ce'
    if args.operator != "mean" and args.acquisition=="adv_train": var_seed += '_{}'.format(args.operator)
    if args.knn_lab: var_seed += '_knn_lab'
    if args.bert_score: var_seed += '_bs'
    if args.bert_rep: var_seed += '_br'
    if args.tfidf: var_seed += '_tfidf'

    results_per_iteration_dir = os.path.join(al_config_dir, var_seed)
    create_dir(results_per_iteration_dir)

    # d_pool_dir = os.path.join(FIG_DIR, 'd_pools')
    # create_dir(d_pool_dir)
    # d_pool_dir = os.path.join(d_pool_dir, '{}_{}_{}'.format(args.dataset_name,
    #                                                                   args.model_type,
    #                                                                   args.acquisition))
    # create_dir(d_pool_dir)
    # d_pool_dir = os.path.join(d_pool_dir, var_seed)
    # create_dir(d_pool_dir)

    exp_dir = os.path.join(EXP_DIR, exp_name)
    # create_dir(exp_dir)

    # fig_dir = os.path.join(FIG_DIR, '{}_{}_{}'.format(args.dataset_name, args.model_type, args.acquisition), var_seed)

    # return fig_dir, results_per_iteration_dir
    return results_per_iteration_dir