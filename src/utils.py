import numpy as np
import os
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True


DATASET_PATHS = {
    'skeletal-age': 'data/skeletal-age/',
    'retina': 'data/retina/',
    'mura': 'data/mura/',
    'mimic-cxr': 'data/mimic-cxr/'
}
NUM_CLASSES = {
    'mura': 2,
    'retina': 5,
    'retina-new': 5,
    'mimic-cxr': 2
}


def evaluate(data_loader, net, mode):
    '''
    Evaluates a network on a specific dataset. Returns predictions and confidences in numpy arrays.
    '''
    assert mode in ['baseline', 'confidence_branch', 'outlier_exposure']
    confidences = []
    predictions = []

    print(f'Evaluating model!')
    pbar = tqdm(total=len(data_loader))
    for test_iter, sample in enumerate(data_loader, 0):
        images, labels = sample
        x = images.cuda()
        logits, confidence = net(x)
        pred = torch.argmax(logits, dim=-1).data.cpu().numpy()
        confidence_logits = get_confidence(logits, confidence, mode) # TODO!
        predictions.append(pred == labels.data.cpu().numpy())
        confidences.append(confidence_logits)

        del images, labels, sample, logits, confidence, pred, confidence_logits
        torch.cuda.empty_cache()
        pbar.update()
    pbar.close()

    predictions = np.concatenate(predictions)
    confidences = np.concatenate(confidences)
    return predictions, confidences


def get_confidence(logits, confidence_logits, mode):
    '''
    Returns the confidence of a model on a batch of data.
    '''
    if mode == 'baseline' or mode == 'outlier_exposure':
        confidence = get_confidence_baseline(logits)
    elif mode == 'confidence_branch':
        confidence = get_confidence_devires(confidence_logits)
    else:
        raise NotImplementedError
    return confidence


def get_confidence_baseline(logits):
    '''
    Given the logits from a baseline model, returns the confidence as a numpy array. Confidence is 
    calculated as the max value after softmax is applied. 
    '''
    probability = torch.softmax(logits, dim=-1)
    pred_value, _ = torch.max(probability.data, 1)
    confidence = pred_value.cpu().numpy()
    confidence = np.reshape(confidence, (len(confidence), 1))
    return confidence


def get_confidence_devires(confidence_logits):
    '''
    Given the pre-sigmoid confidence scores of a Confidence Branch model, returns the post-sigmoid 
    confidence as a numpy array. Confidence Branch is from the following paper: 
    https://arxiv.org/pdf/1802.04865.pdf
    '''
    confidence = torch.sigmoid(confidence_logits).data.cpu().numpy()
    return confidence


def open_file(file_path):
    if not os.path.exists(file_path):
        print(file_path)
        raise FileNotFoundError()
    if '.npy' in file_path:
        return Image.fromarray(np.load(file_path)).convert('RGB')
    else:  # jpg
        return Image.open(str(file_path)).convert('RGB')
