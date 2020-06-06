# pytorch-nli
This repository aims for providing all the baseline models for Natural Language Inference(NLI) task. The main objective of this repository is to provide extensible code structure and framework for working on NLI tasks. This repository can be taken and used to train various combination of models and datasets. Evaluation on trained model and pre-trained model can also be done. This repository is written using Pytorch and Torchtext.


#### Supported Options
Model | Dataset
|---|---|
| BiLSTM | SNLI
|   -  | MultiNLI

#### Results
Model | snli-validation | snli-test | pretrained-model
----|----|----|----|
`BiLSTM ` | 84.24 | 62.419 | download link |

Model | mnli-dev-matched | mnli-dev-mismatched | pretrained-model
----|----|----|----|
`BiLSTM ` | 70.63 | 62.419 | download link |

## Setup
### Requirement

### Integrated Datasets
1) Stanford Natural Language Inference [[website]](https://nlp.stanford.edu/projects/snli/) [[paper]](https://nlp.stanford.edu/pubs/snli_paper.pdf)
2) Multi-Genre Natural Language Inference [[website]](https://www.nyu.edu/projects/bowman/multinli/)[[paper]](https://cims.nyu.edu/~sbowman/multinli/paper.pdf)

## Training
```shell
  python train.py --dataset=snli --model=bilstm
```

## Evaluation
```shell
  python evaluate.py --dataset=snli --model=bilstm --save_path=save/bilstm-snli-model.pt
```
Evaluation script will print following metrics for the model(bilstm) and dataset(snli):
#### Test Accuracy
Test accuracy of save model on requested dataset. Additionally, It will give the label-wise accuracy for that pair of model and dataset
```shell
Validation_accuracy = 80.035, Test loss = 0.591, Test accuracy = 78.746
label: entailment           Accuracy: 84.175
label: contradiction        Accuracy: 79.147
label: neutral              Accuracy: 72.662
```

#### Confusion Matrix
Print the confusion matrix for the model prediction. It will help in calculating precision-recall and other metrics
```shell
+----------------------+-----------------+--------------------+--------------+--------+
|        Table         | entailment-pred | contradiction-pred | neutral-pred | total  |
+----------------------+-----------------+--------------------+--------------+--------+
|  entailment-actual   |      2835.0     |       180.0        |    353.0     | 3368.0 |
| contradiction-actual |      223.0      |       2562.0       |    452.0     | 3237.0 |
|    neutral-actual    |      474.0      |       406.0        |    2339.0    | 3219.0 |
|        total         |      3532.0     |       3148.0       |    3144.0    | 9824.0 |
+----------------------+-----------------+--------------------+--------------+--------+
```

## Evaluate on Pre-trained models

## Contribution
