# code from
# (1) https://github.com/drimpossible/Sampling-Bias-Active-Learning

import gc
import os
import sys

import torch

import numpy as np
from sklearn.metrics import calinski_harabaz_score, f1_score, confusion_matrix, matthews_corrcoef
# from numba import prange
from transformers.data.metrics import simple_accuracy, acc_and_f1, pearson_and_spearman
from sklearn.preprocessing import LabelBinarizer

from utilities.torch_utils import logit_mean

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def acc_and_f1_macro(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name in ["imdb"]:
        return acc_and_f1_macro(preds, labels)
    else:
        raise KeyError(task_name)

def softmax(x):
    assert(len(x.shape)==2)
    e_x =  np.exp(x*1.0)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

# Uncertainty Metrics

def entropy_(y_prob, return_vec=False):
    ent = -1.0 * np.sum(np.multiply(y_prob, np.log(y_prob + np.finfo(float).eps)), axis=1) / np.log(2)
    ent_mean = np.mean(ent)
    ent_var = np.var(ent)
    if return_vec:
        return ent
    return {'mean': float(round(ent_mean,3)), 'var': float(round(ent_var,3))}
    # return float(ent_mean), float(ent_var)

def random(y_prob, return_vec=True):
    uncertainty = np.random.rand(y_prob.shape[0])
    return uncertainty

if __name__ == "__main__":
    num_classes = 3
    n_points = 5
    class_vec = np.arange(num_classes)
    y_label = np.random.choice(class_vec, n_points)
    y_pred_raw = np.random.rand(n_points, num_classes)
    y_pred = softmax(y_pred_raw)

    ones = np.ones(n_points)*1.0
    assert(np.allclose(np.sum(y_pred,axis=1),ones))
    enc = LabelBinarizer()
    enc.fit(y_label)

    #Test the functions given in this script

    print("Expected Calibration Error (Calibration, refinement, Brier Multiclass): ",ece_(y_pred, y_label, n_bins=15))
    # print("F1 Score: ", f1(y_pred, y_label))

    # print("Accuracy: ", accuracy(y_pred, y_label))
    print("Entropy (mean, var): ", entropy_(y_pred))

def uncertainty_metrics(logits, y_label, pool=False, num_classes=None):
    """

    :param logits:
    :param y_label:
    :param pool:
    :return:
    """
    if isinstance(logits, list):
        # logits = list of tensors from MC dropout
        y_pred_raw_ = torch.stack(logits, 1)
        y_pred_raw = logit_mean(y_pred_raw_, dim=1, keepdim=False)
        y_pred = softmax(y_pred_raw.cpu().numpy())
    elif logits.shape.__len__() == 2:
        # logits = tensor
        y_pred = softmax(logits.cpu().numpy())

    if num_classes is None:
        num_classes = len(set(y_label))
    entr = entropy_(y_pred)
    uncertainty_metrics_dict = {"entropy": entr}

    if pool:
        uncertainty_metrics_dict.update({'prob': [list(map(float, y)) for y in y_pred]})
    return uncertainty_metrics_dict