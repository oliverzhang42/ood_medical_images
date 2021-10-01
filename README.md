# Out of Distribution Experiments
This is the code repository for the paper 
["Out of Distribution Detection for Medical Images"](https://link.springer.com/chapter/10.1007/978-3-030-87735-4_10).

## Setup

### Libraries
The pip dependencies can be found under `setup/requirements.txt`. They are as follows:
```
matplotlib==3.3.3
numpy==1.19.5
omegaconf==2.0.6
pandas==1.2.0
scikit-learn==0.24.1
seaborn==0.11.1
torch==1.7.1
torchvision==0.8.2
```

### Data
We use the following datasets: Diabetic Retinopathy [[1]](#1), MIMIC-CXR [[2]](#2), MURA [[3]](#3), 
and RSNA Bone Age [[4]](#4). For each dataset, we resize the images to 224x224 and normalize pixel
values to between 0 and 1. We also turn each dataset into a classification problem and provide the
csv. To download each dataset, please check each link.

## Running the Code
There are a total of four scripts under `scripts`: three will train new models and save them under
`checkpoints`. Specifically, `train_baseline.py` will train a baseline or Maximum Softmax 
Probability (MSP) model [[5]](#5), `train_cb.py` will train a Confidence Branch model [[6]](#6), 
and `train_oe.py` will train an Outlier Exposure model [[7]](#7). Configs which lay out the 
training run's hyperparameters can be found under `config`. After training is finished,
 `fpr_volatility.py` takes in a path to an experiment and will calculate the volatility of the 
 FPR at 95 TPR metric.

### Example
To run a model, we could do

```
python3 train_baseline.py config/baseline/baseline_mimic.yaml
```

Then, the script would train and put the results under `checkpoints/baseline_retina`. Finally, we
could run

```
python3 fpr_volatility.py checkpoints/baseline_retina
```

to get the volatility of the FPR at 95 TPR metric over all runs in the experiment.

### Important Config Parameters
- `num_models` represents how many times we want to run the experiment. By default, the configs run 
each experiment five times.
- `early_stop` defines our patience when deciding when to stop early.
- `eval_start` represents how many epochs to train for before starting to track OOD performance 
and doing early stopping.

## Citations
<a id="1">[1]</a> 
https://www.kaggle.com/c/diabetic-retinopathy-detection

<a id="2">[2]</a> 
https://physionet.org/content/mimic-cxr/2.0.0/

<a id="3">[3]</a> 
https://stanfordmlgroup.github.io/competitions/mura/

<a id="4">[4]</a> 
https://www.kaggle.com/kmader/rsna-bone-age

<a id="5">[5]</a> 
https://arxiv.org/pdf/1610.02136.pdf

<a id="6">[6]</a> 
https://arxiv.org/pdf/1802.04865.pdf

<a id="7">[7]</a> 
https://arxiv.org/abs/1812.04606
