import torch
import torch.optim as O
import torch.nn as nn

import datasets
import models

import datetime

from prettytable import PrettyTable
from utils import *
from pdb import set_trace

class Evaluate():
	def __init__(self):
		print("program execution start: {}".format(datetime.datetime.now()))
		self.args = parse_args()
		self.device = get_device(self.args.gpu)
		self.logger = get_logger(self.args, "evaluate")

		self.dataset_options = {
									'batch_size': self.args.batch_size, 
									'device': self.device
								}
		self.dataset = datasets.__dict__[self.args.dataset](self.dataset_options)
		
		self.validation_accuracy, self.model_options, model_dict = self.load_model()
		self.model = models.__dict__[self.args.model](self.model_options)
		self.model.to(self.device)
		self.model.load_state_dict(model_dict)
		
		self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
		self.test_accuracy = -1	
		print("resource preparation done: {}".format(datetime.datetime.now()))

	def load_model(self):
		model = torch.load('{}/{}/{}/best-{}-{}-params.pt'.format(self.args.results_dir, self.args.model, self.args.dataset, self.args.model, self.args.dataset))
		return model['accuracy'], model['options'], model['model_dict']

	def print_confusion_matrix(self, labels, confusion_matrix):
		table = PrettyTable()
		table.field_names = ["confusion-matrix"] + ["{}-pred".format(label) for label in labels] + ["total"]
		row_label = ["{}-actual".format(label) for label in labels] + ["total"]
	
		row_sum = confusion_matrix.sum(dim=1).view(len(labels), 1)
		confusion_matrix = torch.cat((confusion_matrix, row_sum), dim=1)
		col_sum = confusion_matrix.sum(dim=0).view(1, len(labels)+1)
		confusion_matrix = torch.cat((confusion_matrix, col_sum), dim=0)
		confusion_matrix = confusion_matrix.tolist()
		for i, row in enumerate(confusion_matrix):
			table.add_row([row_label[i]] + [int(count) for count in row] )
		print(table)
		self.logger.info(table)

	def label_wise_accuracy(self, lable_map, confusion_matrix):
		table = PrettyTable()
		table.field_names = ["label", "accuracy"]
		for label, value in lable_map.items():
			acc = round((100. * confusion_matrix[value][value]/confusion_matrix[value].sum()).item(), 3)
			table.add_row([label, acc])
		print(table)
		self.logger.info(table)

	def evaluate(self):
		self.model.eval(); self.dataset.test_iter.init_epoch()
		n_correct, n_total, n_loss = 0, 0, 0
		labels = self.dataset.labels().copy()
		confusion_matrix = torch.zeros(len(labels), len(labels))

		with torch.no_grad():
			for batch_idx, batch in enumerate(self.dataset.test_iter):
				answer = self.model(batch)
				loss = self.criterion(answer, batch.label)

				pred = torch.max(answer, 1)[1].view(batch.label.size())
				for t, p in zip(batch.label, pred):
					confusion_matrix[t.long()][p.long()] += 1

				n_correct += (pred == batch.label).sum().item()
				n_total += batch.batch_size
				n_loss += loss.item()

			test_loss = n_loss/n_total
			test_acc = 100. * n_correct/n_total
			return test_loss, test_acc, confusion_matrix

	def execute(self):
		_, test_acc, confusion_matrix = self.evaluate()
		table = PrettyTable()
		table.field_names = ["data", "accuracy"]
		table.add_row(["validation", round(self.validation_accuracy, 3)])
		table.add_row(["test", round(test_acc, 3)])
		print(table)
		self.logger.info(table)
		lable_map = self.dataset.labels()
		self.label_wise_accuracy(lable_map, confusion_matrix)
		self.print_confusion_matrix(lable_map.keys(), confusion_matrix)
		
task = Evaluate()
task.execute()
