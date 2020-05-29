import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import copy
from evaluate.test_set_evaluator import TestSetEvaluator

import PIL
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from preprocessing.dataset_class import MSIDataset
from evaluate.calibrated_graph import Calibrater
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"

class Evaluator:
		def __init__(self,
								valset,
								netname,
								trainer_skeleton = None,
								checkpoint = None,
								trainloader = None,
								valloader = None,
								tensorboard_dir = None,
								threshold = 0.5
								):
			# Pass Lightning trainer in here
			self.trainer_skeleton = trainer_skeleton
			self.netname = netname
			self.checkpoint = checkpoint
			self.trainloader = trainloader
			self.valloader = valloader
			self.valset = valset

			# Hyperparameters for evaluation
			self.tensorboard_dir = tensorboard_dir
			self.threshold = threshold
			self.batch_size = 128

			# Other evaluation model
			self.calibrater = Calibrater()

			# Helper method
			self.test_set_evaluator = TestSetEvaluator(valset, self.threshold)

		def eval_on_test_set(self, y_true, y_pred):
			return self.test_set_evaluator.eval_all(y_pred = y_pred, y_true = y_true)

		def run_test_set(self):
			column_names = ["model_type", "version", "accuracy_score/picture-level", "auc_score/picture-level", "f1_score/picture-level",
											"recall_score/picture-level", "precision_score/picture-level"]

			for index, checkpoint in enumerate(self.checkpoint[-1:]):
				skeleton = copy.deepcopy(self.trainer_skeleton)
				model = skeleton.load_from_checkpoint(checkpoint_path = checkpoint,
																								trainloader = self.trainloader,
																								valloader = self.valloader)
				model.to(device)

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
					logs = self.evaluate(model, valloader)

					pandas_filename = os.path.join(os.getcwd(), "evaluate", "table")
					if not os.path.isdir(pandas_filename):
						print(">> Creating folder at %s" % pandas_filename)
						os.mkdir(pandas_filename)

					pandas_filename = os.path.join(pandas_filename, test_type + ".csv")
					table = pd.DataFrame(columns = column_names)

					if os.path.isfile(pandas_filename):
						print(">> Loading table from %s" % pandas_filename)
						table = pd.read_csv(pandas_filename)

					logs["model_type"] = self.netname
					logs["version"] = checkpoint

					table = table.append(logs, ignore_index = True)
					table.to_csv(pandas_filename, index = False)


		def evaluate(self, model, valloader, mode = "normal"):
			y_true, y_pred = np.array([]), np.array([])

			model.eval()
			with torch.no_grad():
				for images, labels in tqdm(valloader, total = len(valloader)):
					images, labels = images.cuda(), labels.cuda()
					preds = model(images)
					preds = torch.sigmoid(preds.squeeze())
					y_pred = np.concatenate((y_pred, preds.cpu()))
					y_true = np.concatenate((y_true, labels.cpu()))

			tensorboard_logs = {}

			if mode == "patient":
				patients_table = self.valset.get_patient_status()
				# Check if the order is correct
				labels = patients_table[:, 0].astype(int)
				assert np.array_equal(labels, y_true.astype(int))

				patient_true, patient_pred_real, patient_pred_int = np.array([]), np.array([]), np.array([])
				patients = np.unique(patients_table[:, 1])

				for patient_id in patients:
					idx = np.where(patients_table[:, 1] == patient_id)
					patient_true = np.append(patient_true, y_true[idx[0][0]])
					patient_pred_real = np.append(patient_pred_real, np.sum(y_pred[idx]) / idx[0].shape[0])
					patient_pred_int = np.append(patient_pred_int, sum(y_pred[idx] > 0.5) / idx[0].shape[0])

				auc_score = roc_auc_score(patient_true, patient_pred_real)
				patient_pred_int = (patient_pred_int > self.threshold).astype(int)

				tensorboard_logs = {
					"auc_score/patient-level": auc_score,
					"accuracy_score/patient-level": accuracy_score(patient_true, patient_pred_int),
					"f1_score/patient-level": f1_score(patient_true, patient_pred_int),
					"recall_score/patient-level": recall_score(patient_true, patient_pred_int),
					"precision_score/patient-level": precision_score(patient_true, patient_pred_int)
				}

			auc_score = roc_auc_score(y_true.astype(int), y_pred)

			# Get the label here
			y_pred = (y_pred > self.threshold).astype(int)

			tensorboard_logs.update({
				"auc_score/picture-level": auc_score,
				"accuracy_score/picture-level": accuracy_score(y_true, y_pred),
				"f1_score/picture-level": f1_score(y_true, y_pred),
				"recall_score/picture-level": recall_score(y_true, y_pred),
				"precision_score/picture-level": precision_score(y_true, y_pred)
			})

			return tensorboard_logs

		def calibrated_graph(self, version):
			calibrate_graph_dir = os.path.join(os.getcwd(), "evaluate", "calibrate", self.netname)
			if not os.path.isdir(calibrate_graph_dir):
				os.mkdir(calibrate_graph_dir)

			self.writer = SummaryWriter(os.path.join(calibrate_graph_dir, version))

			for index, checkpoint in enumerate(self.checkpoint):
				print(">> Getting calibrated graph for %s" % checkpoint)
				model = self.trainer_skeleton.load_from_checkpoint(checkpoint_path = checkpoint,
																								trainloader = self.trainloader,
																								valloader = self.valloader)
				model.to(device)

				buf = self.calibrater.get_calibrate_graph(model, self.valloader, checkpoint)
				image = PIL.Image.open(buf)
				image = transforms.ToTensor()(image).squeeze()
				self.writer.add_image(checkpoint, image, index)


			self.writer.close()
