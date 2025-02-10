"""
Customized loss functions to account for permutations of the class labels

Reference: https://github.com/zhengdao-chen/GNN4CD/blob/master/src/losses.py
"""

import math
import numpy as np
import torch

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

def permuteposs(C):
    permutor = Permutor(C)
    permutations = permutor.return_permutations()
    return permutations

class Permutor:
    def __init__(self, C):
        self.row = 0
        self.C = C
        self.collection = np.zeros([math.factorial(C), C])

    def permute(self, arr, l, r):
        if l==r:
            self.collection[self.row, :] = arr
            self.row += 1
        else:
            for i in range(l,r+1):
                arr[l], arr[i] = arr[i], arr[l]
                self.permute(arr, l+1, r)
                arr[l], arr[i] = arr[i], arr[l]

    def return_permutations(self):
        self.permute(np.arange(self.C), 0, self.C-1)
        return self.collection

def compute_loss_multiclass(type, pred, labels, C, permute=True):
    """Compute the loss upto permuations of the labels

    Args:
        type: type of loss. Either "mse" or "ce"
        pred: predicted labels of dimension C
        labels: integer ground truth labels
        C: number of classes
        permute: Enable label permutation (default: True)
    """
    if type=="mse":
        return compute_mse_loss_multiclass(pred=pred, labels=labels, C=C, permute=permute)
    elif type=="ce":
        return compute_ce_loss_multiclass(pred=pred, labels=labels, C=C, permute=permute)
    raise ValueError("Loss type: {} is not supported".format(type))

def compute_ce_loss_multiclass(pred, labels, C, permute=True):
    """Compute the cross-entropy loss upto permuations of the labels

    Args:
        pred: predicted labels of dimension C
        labels: integer ground truth labels
        C: number of classes
        permute: Enable label permutation (default: True)
    """

    criterion = torch.nn.CrossEntropyLoss()
    if permute == False:
        return criterion(pred, labels.type(dtype_l))

    loss = 0
    permutations = permuteposs(C=C)
    for j in range(permutations.shape[0]):
        permuted_labels = torch.from_numpy(permutations[j, labels.cpu().numpy().astype(int)])
        loss_under_perm = criterion(pred, permuted_labels.type(dtype_l))

        if (j == 0):
            loss_single = loss_under_perm
        else:
            loss_single = torch.min(loss_single, loss_under_perm)

    loss += loss_single
    return loss

def compute_mse_loss_multiclass(pred, labels, C, permute=True):
    """Compute the mse loss upto permuations of the labels

    Args:
        pred: predicted labels of dimension C
        labels: integer ground truth labels
        C: number of classes
        permute: Enable label permutation (default: True)
    """

    criterion = torch.nn.MSELoss()
    if permute == False:
        Y = torch.nn.functional.one_hot(labels.type(torch.int64)).type(dtype)
        return criterion(pred, Y)

    loss = 0
    permutations = permuteposs(C=C)
    for j in range(permutations.shape[0]):
        permuted_labels = torch.from_numpy(permutations[j, labels.cpu().numpy().astype(int)])
        Y = torch.nn.functional.one_hot(permuted_labels.type(torch.int64)).type(dtype)
        loss_under_perm = criterion(pred, Y)

        if (j == 0):
            loss_single = loss_under_perm
        else:
            loss_single = torch.min(loss_single, loss_under_perm)

    loss += loss_single
    return loss


def from_scores_to_labels_multiclass(pred):
    labels_pred = np.argmax(pred, axis = 1).astype(int)
    return labels_pred

def _compute_accuracy_helper(pred_labels, labels):
    acc = np.mean(pred_labels == labels)
    return acc

def compute_accuracy_multiclass(pred, labels, C, permute=True):
    """Compute the accuracy upto permuations of the labels

    Args:
        loss_fn: criterion for computing loss
        pred: predicted labels of dimension C
        labels: integer ground truth labels
        C: number of classes
        permute: Enable label permutation (default: True)
    """

    acc = 0
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    pred_labels = from_scores_to_labels_multiclass(pred)
    if permute == False:
        acc = _compute_accuracy_helper(pred_labels, labels)
        acc = (acc - 1/C) / (1 - 1/C)
        return acc

    permutations = permuteposs(C=C)
    for j in range(permutations.shape[0]):
        permuted_labels = permutations[j, labels.astype(int)]
        my_acc = np.count_nonzero((pred_labels == permuted_labels)) / pred_labels.size
        acc_under_perm = _compute_accuracy_helper(pred_labels, permuted_labels)

        if (j == 0):
            acc_single = acc_under_perm
        else:
            acc_single = np.max([acc_single, acc_under_perm])

    acc += acc_single
    # a measure of overlap
    acc = (acc - 1/C) / (1 - 1/C)
    return acc


