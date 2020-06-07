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

		def eval_on_test_set(self, y_true, y_pred, y_pred_label = None):
			return self.test_set_evaluator.eval_all(y_pred = y_pred, y_true = y_true, y_pred_label = y_pred_label)

		def evaluate(self, model, valloader = None):
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