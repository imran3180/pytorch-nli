import torch
import torch.nn as nn
import pdb

__all__ = ['bilstm']

class BiLSTM(nn.Module):
	def __init__(self, options):
		super(BiLSTM, self).__init__()
		self.embedding = nn.Embedding(options['vocab_size'], options['embed_dim'])
		self.projection = nn.Linear(options['embed_dim'], 300)
		self.dropout = nn.Dropout(p = options['dp_ratio'])
		self.lstm = nn.LSTM(300, options['d_hidden'], 3)
		self.relu = nn.ReLU()
		self.out = nn.Sequential(
			nn.Linear(1024, 1024),
			self.relu,
			self.dropout,
			nn.Linear(1024, 1024),
			self.relu,
			self.dropout,
			nn.Linear(1024, 1024),
			self.relu,
			self.dropout,
			nn.Linear(1024, options['out_dim'])
		)
		pass

	def forward(self, batch):
		premise_embed = self.embedding(batch.premise)
		hypothesis_embed = self.embedding(batch.hypothesis)
		premise_proj = self.relu(self.projection(premise_embed))
		hypothesis_proj = self.relu(self.projection(hypothesis_embed))
		encoded_premise, _ = self.lstm(premise_proj)
		encoded_hypothesis, _ = self.lstm(hypothesis_proj)
		premise = encoded_premise.sum(dim = 1)
		hypothesis = encoded_hypothesis.sum(dim = 1)
		combined = torch.cat((premise, hypothesis), 1)
		return self.out(combined)

def bilstm(options):
	return BiLSTM(options)