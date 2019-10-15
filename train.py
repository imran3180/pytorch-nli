import os

import torch
import torch.optim as O
import torch.nn as nn

import datasets
import models

import datetime
import pdb
import torch.nn.functional as F
from tqdm import tqdm

from prettytable import PrettyTable

from utils import get_args, get_device
import time

class Train():
	def __init__(self):
		print("program execution start: {}".format(datetime.datetime.now()))
		self.args = get_args("train")
		self.device = get_device(self.args.gpu)
		self.dataset_options = {
									'batch_size': self.args.batch_size, 
									'device': self.device
								}
		self.dataset = datasets.__dict__[self.args.dataset](self.dataset_options)
		self.model_options = {
									'vocab_size': self.dataset.vocab_size(), 
									'embed_dim': self.args.embed_dim, 
									'out_dim': self.dataset.out_dim(),
									'dp_ratio': self.args.dp_ratio
								}
		self.model = models.__dict__[self.args.model](self.model_options)
		self.model.to(self.device)
		self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
		self.opt = O.Adam(self.model.parameters(), lr = self.args.lr)
		self.best_accuracy = -1
		print("resource preparation done: {}".format(datetime.datetime.now()))

	def save_model(self, current_accuracy):
		if current_accuracy > self.best_accuracy:
			self.best_accuracy = current_accuracy
			torch.save({
				'accuracy': self.best_accuracy,
				'options': self.model_options,
				'model_dict': self.model.state_dict(),
			}, 'save/' + "{}-{}-model.pt".format(self.args.model, self.args.dataset))
		pass
	
	def train(self):
		self.model.train(); self.dataset.train_iter.init_epoch()
		n_correct, n_total, n_loss = 0, 0, 0
		for batch_idx, batch in enumerate(self.dataset.train_iter):
			self.opt.zero_grad()
			answer = self.model(batch)
			loss = self.criterion(answer, batch.label)
			
			n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
			n_total += batch.batch_size
			n_loss += loss.item()
			
			loss.backward(); self.opt.step()
		train_loss = n_loss/n_total
		train_acc = 100. * n_correct/n_total
		return train_loss, train_acc

	def validate(self):
		self.model.eval(); self.dataset.dev_iter.init_epoch()
		n_correct, n_total, n_loss = 0, 0, 0
		with torch.no_grad():
			for batch_idx, batch in enumerate(self.dataset.dev_iter):
				answer = self.model(batch)
				loss = self.criterion(answer, batch.label)
				
				n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
				n_total += batch.batch_size
				n_loss += loss.item()

			val_loss = n_loss/n_total
			val_acc = 100. * n_correct/n_total
			return val_loss, val_acc

	def execute(self):
		for epoch in range(self.args.epochs):
			start = time.time()
			train_loss, train_acc = self.train()
			val_loss, val_acc = self.validate()
			if self.args.save_model:
				self.save_model(val_acc)
			print("time taken: {}   epoch: {}   Training loss: {}   Training Accuracy: {}   Validation loss: {}   Validation loss: {}".format(
				round(time.time()-start, 2), epoch, round(train_loss, 3), round(train_acc, 3), round(val_loss, 3), round(val_acc, 3)
			))


task = Train()
task.execute()