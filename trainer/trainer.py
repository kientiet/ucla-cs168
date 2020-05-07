import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class TrainerSkeleton(pl.LightningModule):
    def __init__(self,
                trainloader,
                valloader):
        super(TrainerSkeleton, self).__init__()
        self.trainloader = trainloader
        self.valloader = valloader

        # Define loss
        '''
            This will eventually be same as nn.BCELoss with nn.Sigmoid.
            Here, I will use CrossEntropyLoss in case later on, the num_classes
            is expanded then I don't have to change the loss function            
            TODO: Explore different cross-entropy
        '''
        self.criterion = nn.CrossEntropyLoss()

        # learning rate logs
        self.logs = []

    def forward(self):
        pass

    def configure_optimizers(self):
        pass

    def loss_func(self, logits, labels):
        return self.criterion(logits, labels)

    def total_epoch(self):
        return self.epoch_per_cycle * self.num_cycle
    
    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        logits = self.forward(images)
        loss = self.loss_func(logits, labels)
        return {"loss": loss}

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        '''
            TODO: Add warm start
        '''
        # update params
        optimizer.step()
        self.scheduler.step()
        optimizer.zero_grad()        
    
    def on_batch_end(self):
        self.logs.append(self.optimizer.param_groups[0]["lr"])

    def validation_step(self, train_batch, batch_idx):
        images, labels = train_batch
        preds = self.forward(images)
        val_loss = self.criterion(preds, labels)
        preds = torch.argmax(F.softmax(preds, dim = 1), dim = 1)
        return {"confusion_matrix": confusion_matrix(y_true = labels.cpu().numpy(), y_pred = preds.cpu().numpy()),
            "val_loss": val_loss}
    
    def validation_epoch_end(self, outputs):
        '''
            TODO: print this out after epoch
        '''
        # tn, fp, fn, tp
        evaluation = [matrix["confusion_matrix"].ravel() for matrix in outputs]
        evaluation = np.sum(evaluation, axis = 0)

        # Calculate the metrics
        metrics = {
            "true_positive": evaluation[-1],
            "true_negative": evaluation[0],
            "false_positive": evaluation[1],
            "false_negative": evaluation[2]
        }

        logs = {
            "accuracy": metrics["true_positive"] / np.sum(evaluation) * 100,
            "precision": metrics["true_positive"] / (metrics["true_positive"] + metrics["false_positive"]),
            "recall": metrics["true_positive"] / (metrics["true_positive"] + metrics["false_negative"])
        }
        logs["f1_score"] = 2 * logs["precision"] * logs["recall"] / (logs["precision"] + logs["recall"])
        return logs
    
    def train_dataloader(self):
        return self.trainloader
    
    def val_dataloader(self):
        return self.valloader
    
    def get_channel(self):
        images, labels = next(iter(self.trainloader))
        _, channels, _, _ = images[0]
        return channels
    
    def show_lr(self):
        plt.plot(range(len(self.logs)), self.logs)
        plt.show()