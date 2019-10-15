# pytorch-nli
Evaluation and comparison of various models on nli(SNLI and MNLI dataset).

Model | SNLI | MNLI 
----|----|----|
`bilstm model` | 73.463 | - |
* table contains the test accuracy

This repository is written using Pytorch and Torchtext.

## Objective
The main objective of this repository is to provide extensible code structure and framework for working on NLI tasks. This repository can be taken and used to train various different models with minimal changes in the code.

## Setup
### Requirement

## Dataset
1) Stanford Natural Language Inference ([SNLI](https://nlp.stanford.edu/projects/snli/))
2) Multi-Genre Natural Language Inference ([MultiNLI](https://www.nyu.edu/projects/bowman/multinli/))

## Training
```shell
  python train.py --dataset=snli --model=lstm
```
### Parameters

## Evaluation
```shell
  python evaluate.py --dataset=snli --model=lstm --save_path=save/bilstm-snli-model.pt
```
Evaluation script will print following metrics:
#### Test Accuracy
Test accuracy of save model on requested dataset. Additionally, It will give the label-wise accuracy for that pair of model and dataset
#### Confusion Matrix
Print the confusion matrix for the model prediction. It will help in calculating precision-recall and other metrics
#### Examples
For every combination of prediction and actual label it will print the top-k examples from the dataset. This will help in analyzing the model performance. 

## Supported Options

## Contribution