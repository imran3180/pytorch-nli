# pytorch-nli
This repository aims for providing all the baseline models for Natural Language Inference(NLI) task. The main objective of this repository is to provide extensible code structure and framework for working on NLI tasks. This repository can be taken and used to train various combination of models and datasets. Evaluation on trained model and pre-trained model can also be done. This repository is written using Pytorch and Torchtext.


#### Supported Options
Model | Dataset
|---|---|
| BiLSTM | SNLI [[Website]](https://nlp.stanford.edu/projects/snli/) [[Paper]](https://nlp.stanford.edu/pubs/snli_paper.pdf)
|   -  | MultiNLI [[Website]](https://www.nyu.edu/projects/bowman/multinli/) [[Paper]](https://cims.nyu.edu/~sbowman/multinli/paper.pdf)

#### Results
Model | snli-validation | snli-test | pretrained-model
----|----|----|----|
`BiLSTM ` | 84.24 | 62.419 | download link |

Model | multinli-dev-matched | multinli-dev-mismatched | pretrained-model
----|----|----|----|
`BiLSTM ` | 70.63 | 62.419 | download link |

## Setup

## Training
```shell
  # Training a BiLSTM model on snli dataset
  python train.py -m bilstm -d snli
  
  # Training a BiLSTM model on multinli dataset
  python train.py -m bilstm -d multinli
```

## Evaluation
```shell
  # Evaluating a trained BiLSTM model on snli dataset
  python evaluate.py -m bilstm -d snli
  
  # Evalauting a trained BiLSTM model on multinli dataset
  python evaluate.py -m bilstm -d multinli
```

Evaluation script will print and log(evaluation.log can be found in results_dir) following metrics:
#### Accuracy
```ruby
+------------+----------+
|            | accuracy |
+------------+----------+
| validation |  77.271  |
|    test    |  77.412  |
+------------+----------+
```

#### Label wise Accuracy
```ruby
+---------------+----------+
|     label     | accuracy |
+---------------+----------+
|   entailment  |  85.986  |
| contradiction |  73.525  |
|    neutral    |  72.352  |
+---------------+----------+
```

#### Confusion Matrix
```ruby
+----------------------+-----------------+--------------------+--------------+-------+
|   confusion-matrix   | entailment-pred | contradiction-pred | neutral-pred | total |
+----------------------+-----------------+--------------------+--------------+-------+
|  entailment-actual   |       2896      |        121         |     351      |  3368 |
| contradiction-actual |       358       |        2380        |     499      |  3237 |
|    neutral-actual    |       589       |        301         |     2329     |  3219 |
|        total         |       3843      |        2802        |     3179     |  9824 |
+----------------------+-----------------+--------------------+--------------+-------+
```

## Evaluate on Pre-trained models

## Contribution
