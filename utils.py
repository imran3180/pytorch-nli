import torch
from argparse import ArgumentParser

def training_params():
	parser = ArgumentParser(description='PyTorch/torchtext NLI Tasks - Training')
	parser.add_argument('--dataset', type=str, default='snli')
	parser.add_argument('--model', type=str, default='bilstm')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--embed_dim', type=int, default=300)
	parser.add_argument('--d_hidden', type=int, default=512)
	parser.add_argument('--dp_ratio', type=int, default=0.2)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--lr', type=float, default=.001)
	parser.add_argument('--combine', type=str, default='cat')
	parser.add_argument('--save_model', action='store_false', default=True)
	args = parser.parse_args()
	return args

def evaluate_params():
	parser = ArgumentParser(description='PyTorch/torchtext NLI Tasks - Evaluation')
	parser.add_argument('--dataset', type=str, default='snli')
	parser.add_argument('--model', type=str, default='bilstm')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--save_path', type=str, default = "save/bilstm-snli-model.pt")
	args = parser.parse_args()
	return args

def get_args(mode):
	if mode == "train":
		return training_params()
	elif mode == "evaluate":
		return evaluate_params()

def get_device(gpu_no):
	if torch.cuda.is_available():
		torch.cuda.set_device(gpu_no)
		return torch.device('cuda:{}'.format(gpu_no))
	else:
		return torch.device('cpu')