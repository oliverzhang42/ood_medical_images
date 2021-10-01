import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_baseline(logits, labels):
    '''
    Given the logits and the labels, returns the cross entropy loss.
    '''
    task_loss = nn.CrossEntropyLoss()(logits, labels)
    return task_loss


def get_loss_confidence_branch(logits, confidence_logits, labels, hint_rate, lmbda):
    '''
    Returns the losses associated with the confidence branch method. 
    See paper: https://arxiv.org/pdf/1802.04865.pdf
    '''
    task_probs = F.softmax(logits, dim=-1)
    confidence_prob = torch.sigmoid(confidence_logits)
    _, num_classes = logits.size()
    one_hot_labels = nn.functional.one_hot(labels, num_classes=num_classes).float()

    # Make sure we don't have any numerical instability
    eps = 1e-12
    task_probs = torch.clamp(task_probs, 0. + eps, 1. - eps)
    confidence_prob = torch.clamp(confidence_prob, 0. + eps, 1. - eps)

    mask = (torch.rand_like(confidence_prob) < hint_rate).float()
    conf = confidence_prob * mask + (1 - mask)

    pred_new = task_probs * conf.expand_as(task_probs) + one_hot_labels * (1 - conf.expand_as(one_hot_labels))
    pred_new = torch.log(pred_new)
    cross_entropy_loss = F.nll_loss(pred_new, labels)

    confidence_loss = torch.mean(-torch.log(confidence_prob))
    total_loss = cross_entropy_loss + (lmbda * confidence_loss)
    return total_loss


def get_loss_outlier_exposure(logits, ood_logits, labels, beta):
    '''
    Given the ood_logits, returns the outlier_exposure_loss. 
    See paper: https://arxiv.org/abs/1812.04606
    '''
    task_loss = get_loss_baseline(logits, labels)
    pred = F.softmax(ood_logits, dim=1)
    aux_loss = torch.max(pred, dim=1).values.mean()
    total_loss = task_loss + beta*aux_loss
    return total_loss