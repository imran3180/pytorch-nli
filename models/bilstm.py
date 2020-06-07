import torch
import torch.nn as nn
from pdb import set_trace

__all__ = ['bilstm']

class BiLSTM(nn.Module):
	def __init__(self, options):
		super(BiLSTM, self).__init__()
		self.embed_dim = 300 # 300 is fix because we are initilizing with glove.6B.300d
		self.hidden_size = options['d_hidden']
		self.directions = 2
		self.num_layers = 2
		self.concat = 4
		self.device = options['device']
		self.embedding = nn.Embedding.from_pretrained(torch.load('.vector_cache/{}_vectors.pt'.format(options['dataset'])))
		self.projection = nn.Linear(self.embed_dim, self.hidden_size)
		self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers,
									bidirectional = True, batch_first = True, dropout = options['dp_ratio'])
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p = options['dp_ratio'])

		self.lin1 = nn.Linear(self.hidden_size * self.directions * self.concat, self.hidden_size)
		self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.lin3 = nn.Linear(self.hidden_size, options['out_dim'])

		for lin in [self.lin1, self.lin2, self.lin3]:
			nn.init.xavier_uniform_(lin.weight)
			nn.init.zeros_(lin.bias)

		self.out = nn.Sequential(
			self.lin1,
			self.relu,
			self.dropout,
			self.lin2,
			self.relu,
			self.dropout,
			self.lin3
		)

	def forward(self, batch):
		premise_embed = self.embedding(batch.premise)
		hypothesis_embed = self.embedding(batch.hypothesis)

		premise_proj = self.relu(self.projection(premise_embed))
		hypothesis_proj = self.relu(self.projection(hypothesis_embed))

		h0 = c0 = torch.tensor([]).new_zeros((self.num_layers * self.directions, batch.batch_size, self.hidden_size)).to(self.device)

		_, (premise_ht, _) = self.lstm(premise_proj, (h0, c0))
		_, (hypothesis_ht, _) = self.lstm(hypothesis_proj, (h0, c0))
		
		premise = premise_ht[-2:].transpose(0, 1).contiguous().view(batch.batch_size, -1)
		hypothesis = hypothesis_ht[-2:].transpose(0, 1).contiguous().view(batch.batch_size, -1)

		combined = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis), 1)
		return self.out(combined)

def bilstm(options):
	return BiLSTM(options)