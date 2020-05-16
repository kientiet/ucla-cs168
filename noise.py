import cv2
import os
import torch
import skimage
from skimage.io import imsave, imread
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.load_dataset import raw_to_dict

import imageio.core.util

def ignore_warnings(*args, **kwargs):
    pass

imageio.core.util._precision_warn = ignore_warnings


seed = 42

def generate_noise(images,  type_noise):
    output_dir = os.path.join(os.getcwd(), "evaluate", "data", type_noise)
    print(output_dir)
    for index, (image, label) in tqdm(enumerate(images), total = len(images)):
        image = image[0].numpy().transpose(1, 2, 0)
        image = skimage.util.random_noise(image, mode = type_noise, seed = seed)
        filename = type_noise + "-" + \
                str(int(label.numpy()[0])) + "-" + \
                str(index) +  ".jpg" 

        imsave(os.path.join(output_dir, filename), image)
        # image = imread(os.path.join(output_dir, filename))
        # print(image.shape)
        # plt.imshow(image)
        # plt.show()
        # break
    
trainset, valset = raw_to_dict(data_mode = "small")

batch_size = 1
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 4)
valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, num_workers = 4)

# generate_noise(valloader, "gaussian")
generate_noise(valloader, "poisson")
generate_noise(valloader, "salt")
generate_noise(valloader, "pepper")
# generate_noise(valloader, "s&p")
generate_noise(valloader, "speckle")