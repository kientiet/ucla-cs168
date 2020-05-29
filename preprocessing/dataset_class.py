import torch
import torchvision
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

cat2int = {
	"mss": 1,
	"msimut": 0
}

class MSIDataset(Dataset):
	def __init__(self, dataset, data_dir, data_mode):
		# Dataset will have form (ImageName, label)
		self.dataset = dataset
		self.data_dir = data_dir
		self.data_mode = data_mode
		self.int_labels = None
		self.str_to_int()

		# Necessary transformation
		self.to_tensor = transforms.ToTensor()
		# self.normalize = transforms.Normalize((0.7265438 , 0.5136047 , 0.69041806), (0.15620758, 0.20168874, 0.14780489))
		# self.normalize = transforms.Normalize((0.7265438 , 0.5136047 , 0.69041806), (0.15620758, 0.20168874, 0.14780489))

	def __getitem__(self, index):
		folder_dir = self.dataset[index, 2]

		image = Image.open(folder_dir)
		image = image.convert('RGB')

		# Apply transformation
		image_to_tensor = self.to_tensor(np.array(image))
		# image_to_tensor = self.normalize(image_to_tensor)

		int_label = np.array(self.int_labels[index])
		int_label = torch.from_numpy(int_label).float()

		return image_to_tensor, int_label

	def __len__(self):
		return len(self.dataset)

	def get_patient_data(self, patient_id):
		return np.where(self.dataset[:, -1] == patient_id)

	def get_patient_status(self):
		return np.column_stack((self.int_labels, self.dataset[:, -1]))

	def get_patients(self):
		return np.unique(self.dataset[:, -1])

	def str_to_int(self):
		## This will convert string labels to int number
		self.int_labels = np.vectorize(cat2int.get)(self.dataset[:, 1])

class AugmentDataSet(Dataset):
	def __init__(self, dataset, data_dir):
		# Dataset will have form (ImageName, label)
		self.dataset = dataset
		self.data_dir = data_dir
		self.to_tensor = transforms.ToTensor()

	def __getitem__(self, index):
		image_name = self.dataset[index]

		folder_dir = ""
		for data_dir in self.data_dir:
			if os.path.isfile(os.path.join(data_dir, image_name)):
				folder_dir = os.path.join(data_dir, image_name)
				break

		image = Image.open(folder_dir)
		image = image.convert('RGB')
		image_to_tensor = self.to_tensor(image)
		return image_to_tensor

	def __len__(self):
		return len(self.dataset)


class DummyDataset(Dataset):
	def __init__(self, dataset):
		super().__init__()
		self.dataset = dataset
		self.to_tensor = transforms.ToTensor()

	def __getitem__(self, index):
	 image_dir = self.dataset[index]

	 image = Image.open(image_dir)
	 image = image.convert("RGB")
	 image_to_tensor = self.to_tensor(image)
	 return image_to_tensor

	def __len__(self):
		return len(self.dataset)
