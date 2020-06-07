# pytorch-nli
This repository aims for providing all the baseline models for Natural Language Inference(NLI) task. The main objective of this repository is to provide extensible code structure and framework for working on NLI tasks. This repository can be taken and used to train various combination of models and datasets. Evaluation on trained model and pre-trained model can also be done. This repository is written using Pytorch and Torchtext.


#### Supported Options
Model | Dataset
|---|---|
| BiLSTM | SNLI [[Website]](https://nlp.stanford.edu/projects/snli/) [[Paper]](https://nlp.stanford.edu/pubs/snli_paper.pdf)
|   -  | MultiNLI [[Website]](https://www.nyu.edu/projects/bowman/multinli/) [[Paper]](https://cims.nyu.edu/~sbowman/multinli/paper.pdf)

#### Results
Model | snli-validation | snli-test |
----|----|----|
`BiLSTM ` | 84.515 | 84.151 |

Model | multinli-dev-matched | multinli-dev-mismatched |
----|----|----|
`BiLSTM ` | 71.075 | 70.555 |

## Setup
#### conda:
```shell
conda env create -f environment.yml

# To download the english tokenizer from SpaCy
python -m spacy download en
```
#### pip:
```shell
pip install -r requirements.txt

# To download the english tokenizer from SpaCy
python -m spacy download en
```


## Training
```shell
  # Training a BiLSTM model on snli dataset
  python train.py -m bilstm -d snli
  
  # Training a BiLSTM model on multinli dataset
  python train.py -m bilstm -d multinli
```
* Training script will create log, training.log which can be found in results_dir(default: results).
* In the first run it will take quite bit time(approx: 30 mins) because the script will download the dataset and Glove word vector. For the subsequent runs it will use the saved dataset and word vector so it will be fast.
## Evaluation
```shell
  # Evaluating a trained BiLSTM model on snli dataset
  python evaluate.py -m bilstm -d snli
  
  # Evalauting a trained BiLSTM model on multinli dataset
  python evaluate.py -m bilstm -d multinli
```
* Evaluation script will print and log(evaluation.log can be found in results_dir) following metrics:
#### Accuracy
```ruby
+------------+----------+
|    data    | accuracy |
+------------+----------+
| validation |  84.515  |
|    test    |  84.151  |
+------------+----------+
```
#### Label wise Accuracy
```ruby
+---------------+----------+
|     label     | accuracy |
+---------------+----------+
|   entailment  |  86.876  |
| contradiction |  85.326  |
|    neutral    |  80.118  |
+---------------+----------+
```
#### Confusion Matrix
```ruby
+----------------------+-----------------+--------------------+--------------+-------+
|   confusion-matrix   | entailment-pred | contradiction-pred | neutral-pred | total |
+----------------------+-----------------+--------------------+--------------+-------+
|  entailment-actual   |       2926      |         93         |     349      |  3368 |
| contradiction-actual |       166       |        2762        |     309      |  3237 |
|    neutral-actual    |       355       |        285         |     2579     |  3219 |
|        total         |       3447      |        3140        |     3237     |  9824 |
+----------------------+-----------------+--------------------+--------------+-------+
```

## Evaluate on Pre-trained models
```shell
# Download and use the pretrained BiLSTM model on SNLI dataset
python scripts/download_pretrained.py -m bilstm -d snli
python evaluate.py -m bilstm -d snli

# Download and use the pretrained BiLSTM model on MultiNLI dataset
python scripts/download_pretrained.py -m bilstm -d multinli
python evaluate.py -m bilstm -d multinli
```

## Contribution Note
I am actively maintaining this repository and adding options for more models and dataset. Please create an issue if you are looking for specific model or dataset.

