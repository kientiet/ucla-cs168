import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import copy
import math
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class BaseLineModel(nn.Module):
    def __init__(self, trainloader, valloader, num_classes = 2):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        # Load pretrained model
        self.model = torchvision.models.resnet18(pretrained = True)
        self.output_weight = nn.Linear(1000, num_classes)

        self.finetune_model = nn.Sequential(
            self.model,
            self.output_weight
        )

    def forward(self, inputs):
        outputs = self.finetune_model(inputs)
        return outputs
    
    def lr_finder(self, loss_func):
        model = copy.deepcopy(self.finetune_model)

        beta = 0.95
        # Range of learning rate
        min_lr, max_lr = 1e-8, 10. 
        lr = min_lr

        numTimes = len(self.trainloader) - 1
        constJump = (max_lr / min_lr) ** (1 / numTimes)

        optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.8)

        losses = []
        log_lrs = []
        avg_loss, best_loss, batch_num = 0., 0., 0.
        
        for images, labels in tqdm(self.trainloader, total = len(self.trainloader)):
            model.train()
            batch_num += 1
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = loss_func(predictions, labels)

            optimizer.zero_grad()
            ## Smooth the curve
            avg_loss = beta * avg_loss + (1 - beta) * loss.cpu().item()
            smooth_loss = avg_loss / (1 - beta ** batch_num)
            if batch_num > 1 and smooth_loss > 4 * best_loss: break
            if smooth_loss < best_loss or batch_num == 1: best_loss = smooth_loss

            losses.append(smooth_loss)
            log_lrs.append(math.log10(optimizer.param_groups[0]['lr']))

            loss.backward()
            optimizer.step()

            lr *= constJump        
            optimizer.param_groups[0]['lr'] = lr

        ## Loss vs. learning rate    
        plt.plot(log_lrs[10:-5], losses[10:-5])
        plt.title("Loss vs. learning rate")
        plt.set(xlabel='lr', ylabel='loss')
        plt.show()