'''
	This file is used to load supervised dataset
'''
import os
import numpy as np
from preprocessing.dataset_class import MSIDataset

def load_data(mode, data_type = "CRC_DX"):
	print("\n\n>> Loading the {}set".format(mode))
	msimut_dir = os.path.join(os.getcwd(), "data", data_type, mode, "msimut")
	mss_dir = os.path.join(os.getcwd(), "data", data_type, mode, "mss")
	dataset = np.array([])

	for data_dir, label in zip([msimut_dir, mss_dir], ["msimut", "mss"]):
		assert os.path.isdir(data_dir)
		all_files = np.array(os.listdir(data_dir))
		label_array = np.full(len(all_files), label)
		image_dir = np.full(len(all_files), data_dir + "/")
		combine = np.column_stack((all_files, label_array,
															np.core.defchararray.add(image_dir, all_files)))

		if len(dataset) == 0:
			dataset = combine
		else:
			dataset = np.concatenate((dataset, combine))

	print("Total images are %d\n\n" % len(dataset))
	return dataset, msimut_dir, mss_dir


def load_train(data_type = "CRC_DX"):
	trainset, msimut_dir, mss_dir = load_data(mode = "train", data_type = data_type)

	# Get the patients id
	patient_ids = np.apply_along_axis(get_patient_id, 1, trainset)
	trainset = np.column_stack((trainset, patient_ids))

	return trainset, msimut_dir, mss_dir


def get_patient_id(data_row):
	case_id = data_row[0]
	patient_id = case_id.split("-")[2:5]

	return "-".join(patient_id)


def load_test(data_type = "CRC_DX"):
	valset, msimut_dir, mss_dir = load_data(mode = "test", data_type = data_type)

	# Get the patients id
	patient_ids = np.apply_along_axis(get_patient_id, 1, valset)
	valset = np.column_stack((valset, patient_ids))

	return valset, msimut_dir, mss_dir


def load_train_test(data_type = "CRC_DX", data_mode = "normal"):
	# Get the trainset and transform to Dataset
	trainset, msimut_dir, mss_dir = load_train(data_type = data_type)
	trainset = MSIDataset(trainset, [msimut_dir, mss_dir], data_mode = "train")

	# Get the valset and transform to Dataset
	valset, msimut_dir, mss_dir = load_test(data_type = data_type)
	valset = MSIDataset(valset, [msimut_dir, mss_dir], data_mode = "test")

	return trainset, valset