import numpy as np
import os
from os import listdir
from collections import defaultdict
from tqdm import tqdm
from preprocessing.dataset_class import MSIDataset

max_size = 1000
data_dir = os.path.join(os.getcwd(), "data")

def extract_file_names(data_dir):
    patients = defaultdict(list)
    if not os.path.isdir(data_dir): return

    # Read from the file
    all_files = listdir(data_dir)
    for file_name in tqdm(all_files):
        # Only accept the jpg file
        if ".jpg" in file_name: 
            components = file_name.split("-")
            patient_id = "-".join(components[2:5])
            patients[patient_id].append([file_name])
    
    print(">> Total running {}".format(len(all_files)))

    return patients

def load_raw_data(data_dir, msimut_dir, mss_dir):
    print("\n>> Loading msimut at {}".format(msimut_dir))
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

def raw_to_dict(data_mode = "normal"):
    # Get the name of in file
    msimut_dir = os.path.join(data_dir, "msimut")
    mss_dir = os.path.join(data_dir, "mss")
    if data_mode == "small":
        msimut_dir = os.path.join(data_dir, "msimut_small")
        mss_dir = os.path.join(data_dir, "mss_small")

    # Load patients from files
    msimut_patients, mss_patients = load_raw_data(data_dir, msimut_dir = msimut_dir, mss_dir = mss_dir)
    # Custom split data
    train, val = custom_split(mss_patients = mss_patients, msimut_patients = msimut_patients)
    # Transform train and val
    print("\n>> From dictionary to data instance")
    trainset = transformation(train, mss_patients, msimut_patients)
    valset = transformation(val, mss_patients, msimut_patients)
    print("\n>> Done!")
    return MSIDataset(trainset, data_dir, data_mode = data_mode), MSIDataset(valset, data_dir, data_mode = data_mode)