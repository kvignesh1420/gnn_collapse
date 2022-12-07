"""
Customized loss functions to account for permutations of the class labels

Reference: https://github.com/zhengdao-chen/GNN4CD/blob/master/src/losses.py
"""

import math
import numpy as np
import torch

criterion = torch.nn.CrossEntropyLoss()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

def permuteposs(k):
    permutor = Permutor(k)
    permutations = permutor.return_permutations()
    return permutations

class Permutor:
    def __init__(self, k):
        self.row = 0
        self.k = k
        self.collection = np.zeros([math.factorial(k), k])

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
        self.permute(np.arange(self.k), 0, self.k-1)
        return self.collection


def compute_loss_multiclass(pred, labels, k):
    """Compute the loss upto permuations of the labels

    Args:
        loss_fn: criterion for computing loss
        pred: predicted labels of dimension k
        labels: integer ground truth labels
        k: number of classes
    """

    loss = 0
    permutations = permuteposs(k=k)
    for j in range(permutations.shape[0]):
        permuted_labels = torch.from_numpy(permutations[j, labels.numpy().astype(int)])
        loss_under_perm = criterion(pred, permuted_labels.type(dtype_l))

        if (j == 0):
            loss_single = loss_under_perm
        else:
            loss_single = torch.min(loss_single, loss_under_perm)

    loss += loss_single
    return loss


def from_scores_to_labels_multiclass(pred):
    labels_pred = np.argmax(pred, axis = 1).astype(int)
    return labels_pred

def _compute_accuracy_helper(labels_pred, labels):
    acc = np.mean(labels_pred == labels)
    return acc

def compute_accuracy_multiclass(pred, labels, k):
    """Compute the accuracy upto permuations of the labels

    Args:
        loss_fn: criterion for computing loss
        pred: predicted labels of dimension k
        labels: integer ground truth labels
        k: number of classes
    """

    acc = 0
    pred = pred.detach().numpy()
    labels = labels.detach().numpy()
    pred_labels = from_scores_to_labels_multiclass(pred)
    # print(pred_labels, labels)
    permutations = permuteposs(k=k)
    # print(permutations)
    for j in range(permutations.shape[0]):
        permuted_labels = permutations[j, labels.astype(int)]
        acc_under_perm = _compute_accuracy_helper(pred_labels, permuted_labels)

        if (j == 0):
            acc_single = acc_under_perm
        else:
            acc_single = np.max([acc_single, acc_under_perm])

    acc += acc_single
    # a measure of overlap
    acc = (acc - 1/k) / (1 - 1/k)
    return acc


