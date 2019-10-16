# pytorch-nli
Evaluation and comparison of various models on nli(SNLI and MultiNLI dataset).

Model | SNLI | MultiNLI 
----|----|----|
`bilstm model` | 78.746 | 62.419 |
* table contains the test accuracy

This repository is written using Pytorch and Torchtext.

## Objective
The main objective of this repository is to provide extensible code structure and framework for working on NLI tasks. This repository can be taken and used to train various different models with minimal changes in the code.

## Setup
### Requirement

## Dataset
1) Stanford Natural Language Inference ([SNLI](https://nlp.stanford.edu/projects/snli/))
	* SNLI dataset is supported by torchtext so you don't have to download it explicitly. torchtext dataset will download it locally in the .data/ folder when program will run for the first time.
2) Multi-Genre Natural Language Inference ([MultiNLI](https://www.nyu.edu/projects/bowman/multinli/))
	* MultiNLI dataset is supported by torchtext so you don't have to download it explicitly. torchtext dataset will download it locally in the .data/ folder wheb program will run for the first time.

## Training
```shell
  python train.py --dataset=snli --model=bilstm
```
#### Parameters:
Name | Description | Default Value
---|---|---|
`dataset` | Dataset name | snli
`model` | Model name | bilstm
`gpu` | GPU number, if GPU is present | 0
`batch_size` | Size of each batch for training, validation and test Iterator | 128
`embed_dim` | Embedding size of the word representation | 300
`d_hidden` | Hidden size in lstm | 512
`dp_ratio` | dropout ratio in linear layers | 0.2
`epochs` | # of epochs required for the training | 50
`lr` | Learning rate | 0.001
`combine` | Method to combine the premise and hypothesis | cat

#### Supported Options
Model | Dataset
|---|---|
| lstm | SNLI
|   -  | MultiNLI


### Pre-Trained Model

Model | dataset | Test Accuracy | batch_size, embed_dim, d_hidden, dp_ratio, epochs, lr, combine
---|---|---|---|
`bilstm model` | SNLI | 78.746 | ( 128, 300, 512, 0.2, 50, 0.001, 'cat' )
`bilstm model` | MultiNLI | 62.419 | ( 128, 300, 512, 0.2, 50, 0.001, 'cat' )


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
#### Examples
For every combination of prediction and actual label it will print the top-k examples from the dataset. This will help in analyzing the model performance. 


## Contribution