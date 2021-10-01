import json
import numpy as np
import os


def fpr_change(array):
    num = 0
    for i in range(len(array) - 1):
        if abs(array[i] - array[i+1]) > 0.25:
            num += 1
    return num/(len(array) - 1)


def get_results(experiment_folder):
    results_array = []
    for file_name in os.listdir(experiment_folder):
        path = os.path.join(experiment_folder, file_name)
        if os.path.isdir(path):
            results_path = os.path.join(path, 'results.json')
            with open(results_path, 'r') as f:
                results = json.load(f)
            results_array.append(results)
    return results_array


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to experiment folder')
    args = parser.parse_args()
    
    # Get results
    results_array = get_results(args.path)

    # Get volatility of fpr
    volatility = []
    for results in results_array:
        for dataset in results:
            if not(dataset == 'test_accuracy'):
                v = fpr_change(results[dataset]['fpr_at_95_tpr'])
                volatility.append(v)
    prop = np.average(volatility)

    # Print volatility
    print(f'Some {prop} proportion of the epochs have FPR at 95 TPR changing more than 25%')
