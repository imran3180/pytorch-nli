import os
import sys

import torch

from torchtext.data import Field, Iterator, TabularDataset
from torchtext import datasets
from utils import makedirs

from pdb import set_trace

__all__ = ['snli']

class SNLI():
	def __init__(self, options):
		self.TEXT = Field(lower=True, tokenize='spacy', batch_first = True)
		self.LABEL = Field(sequential=False, unk_token = None, is_target = True)
		
		self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)
		
		self.TEXT.build_vocab(self.train, self.dev)
		self.LABEL.build_vocab(self.train)

		vector_cache_loc = '.vector_cache/snli_vectors.pt'
		if os.path.isfile(vector_cache_loc):
			self.TEXT.vocab.vectors = torch.load(vector_cache_loc)
		else:
			self.TEXT.vocab.load_vectors('glove.840B.300d')
			makedirs(os.path.dirname(vector_cache_loc))
			torch.save(self.TEXT.vocab.vectors, vector_cache_loc)

		self.train_iter, self.dev_iter, self.test_iter = Iterator.splits((self.train, self.dev, self.test), 
																													batch_size=options['batch_size'], device=options['device'])

	def vocab_size(self):
		return len(self.TEXT.vocab)

	def out_dim(self):
		return len(self.LABEL.vocab)

	def labels(self):
		return self.LABEL.vocab.stoi

def snli(options):
	return SNLI(options)