import torch
import torchvision
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

labels_to_int = {
	"mss": 1,
	"msimut": 0
}

test_to_int = {
	"0": 0,
	"1": 1
}

class MSIDataset(Dataset):
	def __init__(self, dataset, data_dir, data_mode = "normal"):
		# Dataset will have form (ImageName, label)
		self.dataset = dataset
		self.data_dir = data_dir
		self.data_mode = data_mode
		self.to_tensor = transforms.ToTensor()
		self.int_labels = None
		self.str_to_int()

	def __getitem__(self, index):
		image_name, label = self.dataset[index]

		# Check whether we run small, big or test
		folder_dir = os.path.join(self.data_dir, label, image_name)
		if self.data_mode == "small":
			folder_dir = os.path.join(self.data_dir, label + "_small", image_name)
		elif self.data_mode == "test":
			folder_dir = os.path.join(self.data_dir, image_name)

		image = Image.open(folder_dir)
		image = image.convert('RGB')
		image_to_tensor = self.to_tensor(image)
		int_label = np.array(self.int_labels[index])
		int_label = torch.from_numpy(int_label).float()
		return image_to_tensor, int_label

	def __len__(self):
		return len(self.dataset)


	def str_to_int(self):
		## This will convert string labels to int number
		if self.data_mode == "test":
			self.int_labels = np.vectorize(test_to_int.get)(self.dataset[:, 1])
		else:
			self.int_labels = np.vectorize(labels_to_int.get)(self.dataset[:, 1])
		# self.int_labels = np.unique(self.dataset[:, 1], return_inverse = True)[1]