import collections
import logging
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelWithLMHead


sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utilities.torch_utils import mutual_information
from utilities.data_loader import get_glue_tensor_dataset
from utilities.preprocessors import processors


from sys_config import CKPT_DIR
from acquisition.acquisition_utils import mutual_information, max_entropy_acquisition_function

logger = logging.getLogger(__name__)

def entropy(logits):
    """
    Entropy-based uncertainty.
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return:
    """

    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det

    entropy_scores_ = max_entropy_acquisition_function(logits_B_K_C)
    return entropy_scores_.cpu().numpy()

def calculate_uncertainty(args, method, logits, annotations_per_it, device, iteration, task=None,
                          oversampling=False,
                          representations=None,
                          candidate_inds=None,
                          labeled_inds=None,
                          discarded_inds=None,
                          original_inds=None,
                          model=None,
                          X_original=None, y_original=None):
    """
    Selects and performs uncertainty-based acquisition.
    :param method: uncertainty-based acquisition function. options:
        - 'least_conf' for least confidence
        - 'margin_conf' for margin of confidence
        - 'ratio_conf' for ratio of confidence
    :param prob_dist: output probability distribution
    :param logits: output logits
    :param annotations_per_it: number of samples (to be sampled)
    :param D_lab: [(X_labeled, y_labeled)] labeled data
    :param D_unlab: [(X_unlabeled, y_unlabeled)] unlabeled data
    :return:
    """
    #
    prob_dist = None

    init_labeled_data = len(labeled_inds)  # before selecting data for annotation
    init_unlabeled_data = len(candidate_inds)

    if method not in ['random']:
        if type(logits) is list and logits != []:
            assert init_unlabeled_data == logits[0].size(0), "logits {}, inital unlabaled data {}".format(
                logits[0].size(0), init_unlabeled_data)
        elif type(logits) != []:
            assert init_unlabeled_data == len(logits)

    if method == 'entropy':
        uncertainty_scores = entropy(logits)
    elif method == 'random':
        pass
    else:
        raise ValueError('Acquisition function {} not implemented yet check again!'.format(method))

    if method == 'random':
        sampled_ind = np.random.choice(init_unlabeled_data, annotations_per_it, replace=False)
    else:
        # find indices with #samples_to_annotate least confident samples = BIGGER numbers in uncertainty_scores list
        sampled_ind = np.argpartition(uncertainty_scores, -annotations_per_it)[-annotations_per_it:]
        if args.reverse:
            sampled_ind = np.argpartition(uncertainty_scores, annotations_per_it)[:annotations_per_it]
    y_lab = np.asarray(y_original, dtype='object')[labeled_inds]
    X_unlab = np.asarray(X_original, dtype='object')[candidate_inds]
    y_unlab = np.asarray(y_original, dtype='object')[candidate_inds]

    labels_list_previous = list(y_lab)
    c = collections.Counter(labels_list_previous)
    stats_list_previous = [(i, c[i] / len(labels_list_previous) * 100.0) for i in c]

    new_samples = np.asarray(X_unlab, dtype='object')[sampled_ind]
    new_labels = np.asarray(y_unlab, dtype='object')[sampled_ind]

    # Mean and std of length of selected sequences
    l = [len(x.split()) for x in new_samples]
    assert type(l) is list, "type l: {}, l: {}".format(type(l), l)
    length_mean = np.mean(l)
    length_std = np.std(l)
    length_min = np.min(l)
    length_max = np.max(l)

    # Percentages of each class
    labels_list_selected = list(np.array(y_unlab)[sampled_ind])
    c = collections.Counter(labels_list_selected)
    stats_list = [(i, c[i] / len(labels_list_selected) * 100.0) for i in c]

    labels_list_after = list(new_labels) + list(y_lab)
    c = collections.Counter(labels_list_after)
    stats_list_all = [(i, c[i] / len(labels_list_after) * 100.0) for i in c]

    assert len(new_samples) == annotations_per_it, 'len(new_samples)={}, annotatations_per_it={}'.format(len(new_samples), annotations_per_it)
    if args.indicator is not None:
        pass
    else:
        assert len(labeled_inds) + len(candidate_inds) + len(discarded_inds) == len(original_inds)

    stats = {'length': {'mean': float(length_mean),
                        'std': float(length_std),
                        'min': float(length_min),
                        'max': float(length_max)},
             'class_selected_samples': stats_list,
             'class_samples_after': stats_list_all,
             'class_samples_before': stats_list_previous}

    return sampled_ind, stats


if __name__ == '__main__':
    # prob_dist = [torch.rand(100, 2), torch.rand(100, 2), torch.rand(100, 2)]
    MC_1 = torch.tensor(np.array([0.5, 0.5]))
    MC_2 = torch.tensor(np.array([0.6, 0.4]))
    MC_3 = torch.tensor(np.array([0.3, 0.7]))

    prob_dist = [MC_1, MC_2, MC_3]
    # test entropy function:

    print()
