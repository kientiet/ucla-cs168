import numpy as np
import os
from os import listdir
from collections import defaultdict
from tqdm import tqdm
from preprocessing.dataset_class import MSIDataset

max_size = 1000
data_dir = os.path.join(os.getcwd(), "data")

# Get the name of in file
msimut_dir = os.path.join(data_dir, "msimut_split/msimut")
mss_dir = os.path.join(data_dir, "mss_split/mss")

def extract_file_names(data_dir):
    patients = defaultdict(list)
    total_iteration = 0
    for i in range(1, max_size + 1):
        all_files = listdir(data_dir + str(i))
        for file_name in all_files:
            # Only accept the jpg file
            if ".jpg" in file_name: 
                total_iteration += 1
                components = file_name.split("-")
                patient_id = "-".join(components[2:5])
                patients[patient_id].append([file_name, i])
        break
    
    print(">> Total running {}".format(total_iteration))

    return patients

def load_raw_data(data_dir):
    print(">> Loading msimut at {}".format(msimut_dir))
    msimut_patients = extract_file_names(msimut_dir)

    print("\n>> Loading msimut at {}".format(mss_dir)) 
    mss_patients = extract_file_names(mss_dir)

    return msimut_patients, mss_patients

def custom_split(mss_patients, msimut_patients, factors = 0.8):
    print("\n>> Splitting the dataset")
    # Load the unique patient_id from the dictionary    
    list_mss_patients = list(mss_patients.keys())
    list_msimut_patients = list(msimut_patients.keys())

    # Split to train and test test in MSS set
    np.random.seed(42)
    train_mss = np.random.choice(list_mss_patients, np.around(factors * len(list_mss_patients)).astype(int))
    val_mss = np.intersect1d(list_mss_patients, train_mss)

    # Split to train and test set in MSI set
    train_msimut = np.random.choice(list_msimut_patients, np.around(factors * len(list_msimut_patients)).astype(int))
    val_msimut = np.intersect1d(list_msimut_patients, train_msimut)

    train = ((train_mss, "mss"), (train_msimut, "msimut"))
    val = ((val_mss, "mss"), (val_msimut, "msimut"))
    return train, val

def transformation(dataset, *patients_table):
    finalize_dataset = np.array([])
    for index, (data, label) in enumerate(dataset):
        table = patients_table[index]
        for patients in data:
            file_names = np.array(table[patients])
            label_vector = np.full(len(file_names), label)
            input_instance = np.vstack((file_names.T, label_vector)).T
            if len(finalize_dataset) == 0:
                finalize_dataset = input_instance
            else: 
                finalize_dataset = np.vstack((finalize_dataset, input_instance))
    
    return finalize_dataset

def raw_to_dict():
    # Load patients from files
    msimut_patients, mss_patients = load_raw_data(data_dir)
    # Custom split data
    train, val = custom_split(mss_patients = mss_patients, msimut_patients = msimut_patients)
    # Transform train and val
    print("\n>> From dictionary to data instance")
    trainset = transformation(train, mss_patients, msimut_patients)
    valset = transformation(val, mss_patients, msimut_patients)
    print("\n>> Done!")
    return MSIDataset(trainset, data_dir), MSIDataset(valset, data_dir)