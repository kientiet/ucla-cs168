import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
from pprint import pprint
from preprocessing.dataset_class import MSIDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score


device = "cuda" if torch.cuda.is_available() else "cpu"

class Evaluator:
		def __init__(self, model, checkpoint, trainloader, valloader):
			# Pass Lightning trainer in here
			self.model = model.load_from_checkpoint(checkpoint_path = checkpoint,
																							trainloader = trainloader,
																							valloader = valloader)
			self.model.to(device)
			self.valloader = valloader
			self.threshold = 0.5
			self.batch_size = 128

		def run_validation(self):
			logs = self.evaluate(self.valloader)
			pprint(logs)

		def run_test_set(self):
			test_dir = os.path.join(os.getcwd(), "evaluate", "data")
			for test_type in os.listdir(test_dir):
				print("\n\n>> Evaluate %s noise" % test_type)
				test_dir_noise = os.path.join(test_dir, test_type)

				print("Extracting the test images in %s" % test_dir_noise)
				dataset = []
				all_files = os.listdir(test_dir_noise)
				for file_name in tqdm(all_files):
						# Only accept the jpg file
						if ".jpg" in file_name:
								components = file_name.split("-")
								dataset.append([file_name, int(components[1])])

				print("Transform to dataset and loader")
				valset = MSIDataset(np.array(dataset), test_dir_noise, data_mode = "test")
				valloader = torch.utils.data.DataLoader(valset, batch_size = self.batch_size, num_workers = 4)

				print("Start evaluating...")
				logs = self.evaluate(valloader)
				pprint(logs)


		def evaluate(self, valloader):
			y_pred, y_true = np.array([]), np.array([])
			self.model.eval()
			with torch.no_grad():
				for images, labels in tqdm(valloader, total = len(valloader)):
					images, labels = images.to(device), labels.to(device)
					preds = self.model(images)
					preds = preds.cpu().numpy().reshape(images.shape[0])
					labels = labels.cpu().numpy().reshape(images.shape[0])

					y_pred = np.concatenate((preds, y_pred))
					y_true = np.concatenate((labels, y_true))

			auc_score = roc_auc_score(y_true, y_pred)

			# Get the label here
			y_pred = (y_pred > self.threshold).astype(int)
			y_true = y_true.astype(int)

			tensorboard_logs = {
				"auc_score": auc_score,
				"accuracy_score": accuracy_score(y_true, y_pred),
				"f1_score": f1_score(y_true, y_pred),
				"recall_score": recall_score(y_true, y_pred),
				"precision_score": precision_score(y_true, y_pred)
			}

			return tensorboard_logs