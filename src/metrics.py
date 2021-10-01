'''
Stores the metrics used in the paper.
'''
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def calc_metrics(preds, labels):
    '''
    Using predictions and labels, return a dictionary containing all novelty detection performance 
    statistics. These metrics conform to how results are reported in the paper 'Enhancing The
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'. Link:
    https://arxiv.org/abs/1706.02690
    '''
    return fpr_at_95_tpr(preds, labels), detection_error(preds, labels), auroc(preds, labels)


def auroc(preds, labels):
    '''
    Calculate and return the area under the ROC curve using model predictions and the binary labels.
    '''
    fpr, tpr, _ = roc_curve(labels, preds)
    return auc(fpr, tpr)


def fpr_at_95_tpr(preds, labels):
    '''
    Returns the false positive rate when the true positive rate is at minimum 95%.
    '''
    fpr, tpr, _ = roc_curve(labels, preds)
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)


def detection_error(preds, labels):
    '''
    Return the misclassification probability when TPR is 95%.
    '''
    fpr, tpr, _ = roc_curve(labels, preds)

    # Get ratio of true positives to false positives
    f2t_ratio = sum(np.array(labels) == 1) / len(labels)
    t2f_ratio = 1 - f2t_ratio

    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x >= 0.95]

    # Calc error for a given threshold (i.e. idx)
    _detection_error = lambda idx: t2f_ratio * (1 - tpr[idx]) + f2t_ratio * fpr[idx]

    # Return the minimum detection error such that TPR >= 0.95
    return min(map(_detection_error, idxs))
