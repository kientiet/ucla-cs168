import torch
import torchvision
from preprocessing.load_dataset import raw_to_dict
import matplotlib.pyplot as plt
import numpy as np

trainset, valset = raw_to_dict()

batch_size = 100
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle = True)

images, labels = next(iter(trainloader))

def imshow(img):
    inp = img.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()

imshow(images[0])