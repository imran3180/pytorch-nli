import sys
from torchtext import data
from torchtext import datasets

import dill
import pdb

__all__ = ['snli']

class SNLI():
	def __init__(self, options):
		self.inputs = data.Field(lower=True, tokenize='spacy', batch_first = True)
		self.answers = data.Field(sequential=False, unk_token = None, is_target = True)
		self.train, self.dev, self.test = datasets.SNLI.splits(self.inputs, self.answers)
		self.inputs.build_vocab(self.train, self.dev)
		self.answers.build_vocab(self.train)
		self.train_iter, self.dev_iter, self.test_iter = data.Iterator.splits((self.train, self.dev, self.test), 
																													batch_size=options['batch_size'], device=options['device'])

	def vocab_size(self):
		return len(self.inputs.vocab)

	def out_dim(self):
		return len(self.answers.vocab)

	def labels(self):
		return self.answers.vocab.stoi

def snli(options):
	return SNLI(options)