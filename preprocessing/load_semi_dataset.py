import os
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from preprocessing.dataset_class import MSIDataset, AugmentDataSet, SemiDataSet
from preprocessing.load_sup_dataset import load_train, load_test


file_extension = ".png"
data_dir = os.path.join(os.getcwd(), "data")
num_copies = 4
np.random.seed(42)


def get_patients_list(trainset):
  patients = trainset[:, [1, -1]]
  patients = np.unique(patients, axis = 0)
  return patients


def split_labeled_and_unlabeled(trainset, semi_ratio):
  labeled_data, unlabeled_data = np.array([]), np.array([])
  for index, label in enumerate(np.unique(trainset[:, 1])):
    images = trainset[trainset[:, 1] == label]
    sample = np.random.choice(np.arange(len(images)), np.ceil(semi_ratio * len(images)).astype(int), replace = False)
    if index == 0:
      labeled_data = images[sample]
      unlabeled_data = images[np.setdiff1d(np.arange(len(images)), sample)]
    else:
      labeled_data = np.concatenate((labeled_data, images[sample]))
      unlabeled_data = np.concatenate((unlabeled_data, images[np.setdiff1d(np.arange(len(images)), sample)]))

    assert len(np.intersect1d(labeled_data[:, 0], unlabeled_data[:, 0])) == 0

  assert labeled_data.shape[0] + unlabeled_data.shape[0] == trainset.shape[0]
  assert np.sum(labeled_data[:, 1] == "msimut") == np.sum(labeled_data[:, 1] == "mss")

  print(">> Total labels sup patients vs unsup patients: %d and %d" % (len(labeled_data), len(unlabeled_data)))
  return labeled_data, unlabeled_data


def get_aug_file(row):
  filename = row[0].split(file_extension)
  return "{}-{}{}".format(filename[0], str(row[1]), file_extension)

def load_semi_train(semi_ratio,
                    data_type,
                    train_batch_size,
                    unsup_batch_size,
                    total_batch_size,
                    aug_dir):

  # Load the trainset
  trainset, msimut_dir, mss_dir = load_train(data_type = data_type)

  labeled_dataset, unlabeled_dataset = split_labeled_and_unlabeled(trainset, semi_ratio)

  assert len(labeled_dataset) + len(unlabeled_dataset) == len(trainset)
  assert len(labeled_dataset) < len(unlabeled_dataset)

  # Shuffle the dataset
  ori_images = np.repeat(unlabeled_dataset, num_copies, axis=0)
  aug_choices = np.repeat(np.arange(num_copies), len(ori_images) // num_copies)
  new_unsup_data = np.column_stack((ori_images, aug_choices))
  np.random.shuffle(new_unsup_data)

  # Get enough dataset
  total_iteration = np.ceil(len(labeled_dataset) / train_batch_size).astype(int)
  total_unsup_amount = unsup_batch_size * total_iteration

  ori_images, aug_choices = new_unsup_data[:total_unsup_amount, :-1], new_unsup_data[:total_unsup_amount, -1]

  # Create complete dataset
  aug_images = np.array([])
  aug_file = np.apply_along_axis(get_aug_file, 1, np.column_stack((ori_images[:, 0], aug_choices)))
  patients = ori_images[:, -1]
  aug_dir = np.full(len(aug_file), aug_dir + "/")
  aug_dir = np.core.defchararray.add(aug_dir, aug_file)
  aug_images = np.column_stack((aug_file, aug_dir, patients))

  ori_images = ori_images[:, [0, 2, 3]]

  print("\n\n>> Total image in sup dataset, ori dataset and aug dataset")
  print(labeled_dataset.shape, ori_images.shape, aug_images.shape)
  assert len(ori_images) == len(aug_images)
  assert np.array_equal(ori_images[:, -1], aug_images[:, -1])

  ori_dataset = AugmentDataSet(ori_images, [msimut_dir, mss_dir])
  aug_dataset = AugmentDataSet(aug_images, [aug_dir])
  labeled_dataset = MSIDataset(labeled_dataset, [msimut_dir, mss_dir], data_mode = "train")

  return labeled_dataset, ori_dataset, aug_dataset

def get_semi_data(train_batch_size, unsup_batch_size,
                  total_batch_size,
                  data_type = "CRC_DX",
                  semi_ratio = 0.05):
  '''
    The data_ratio is to get the training dataset and split follow this ratio between
    labeled and unlabeled data
  '''
  aug_dir = os.path.join(data_dir, data_type, "augmentation")
  trainset, ori_dataset, aug_dataset = load_semi_train(semi_ratio, data_type, train_batch_size, unsup_batch_size, total_batch_size, aug_dir)

  # Get the valset and transform to Dataset
  valset, msimut_dir, mss_dir = load_test(data_type = data_type)
  valset = MSIDataset(valset, [msimut_dir, mss_dir], data_mode = "test")

  return trainset, ori_dataset, aug_dataset, valset