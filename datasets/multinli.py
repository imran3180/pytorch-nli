import os
import sys

import torch

from torchtext.data import Field, Iterator
from torchtext import datasets
from utils import makedirs

from pdb import set_trace

__all__ = ['multinli']

class MultiNLI():
	def __init__(self, options):
		print("preparing the dataset for training...")
		self.TEXT = Field(lower=True, tokenize='spacy', batch_first = True)
		self.LABEL = Field(sequential=False, unk_token = None, is_target = True)
		
		# Since MNLI does not provide public test data
		# self.dev - MultiNLI Matched data
		# self.test - MultiNLI Mismatched data
		# To evaluate your system on the full test set, use the following Kaggle in Class competitions.
		# https://www.kaggle.com/c/multinli-matched-open-evaluation
		# https://www.kaggle.com/c/multinli-mismatched-open-evaluation

		self.train, self.dev, self.test = datasets.MultiNLI.splits(self.TEXT, self.LABEL)
		
		self.TEXT.build_vocab(self.train, self.dev)
		self.LABEL.build_vocab(self.train)

		vector_cache_loc = '.vector_cache/multinli_vectors.pt'
		if os.path.isfile(vector_cache_loc):
			self.TEXT.vocab.vectors = torch.load(vector_cache_loc)
		else:
			self.TEXT.vocab.load_vectors('glove.840B.300d')
			makedirs(os.path.dirname(vector_cache_loc))
			torch.save(self.TEXT.vocab.vectors, vector_cache_loc)
		
		self.train_iter, self.dev_iter, self.test_iter = Iterator.splits((self.train, self.dev, self.test), 
																													batch_size=options['batch_size'], device=options['device'],
																													sort_key = lambda x: len(x.premise), sort_within_batch = False, shuffle = True)

	def vocab_size(self):
		return len(self.TEXT.vocab)

	def out_dim(self):
		return len(self.LABEL.vocab)

	def labels(self):
		return self.LABEL.vocab.stoi

def multinli(options):
	return MultiNLI(options)