import os
import requests
import argparse

from tqdm import tqdm
from pdb import set_trace

# This function is copied from the source
# Source: https://stackoverflow.com/a/39225039
def download_file_from_google_drive(id, destination):
	def get_confirm_token(response):
		for key, value in response.cookies.items():
			if key.startswith('download_warning'):
				return value
		return None

	def save_response_content(response, destination):
		CHUNK_SIZE = 32768

		with open(destination, "wb") as f:
			for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
				if chunk: # filter out keep-alive new chunks
					f.write(chunk)

	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)

	save_response_content(response, destination)

def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='download script for pretrained-models')
	parser.add_argument('--model', '-m', type = str, default = "bilstm", choices=['bilstm'], help = "model name")
	parser.add_argument('--dataset', '-d', type = str, default = "snli", choices=['snli', 'multinli'], help = "dataset name")
	parser.add_argument('--results_dir', type=str, default='results')
	args = parser.parse_args()

	file_ids = {
		"bilstm": {
			"snli": "1m1Kyhcc5CMnyu8qwf6SlteKd9UNb0Hfn",
			"multinli": "1WZZZEHabpu1qWuraAKGrOhdx6V_AdyOr"
		}
	}
	check_folder(f"{args.model}/{args.dataset}")
	download_path = f"{args.model}/{args.dataset}/best-{args.model}-{args.dataset}-params.pt"

	download_file_from_google_drive(file_ids[args.model][args.dataset], download_path)
	print(f" [*] model best-{args.model}-{args.dataset}-params.pt downloaded from the google drive.")
	
