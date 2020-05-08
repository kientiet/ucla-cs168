import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import itertools
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

        # Logging config
        # self.lr_log = [] # learning rate logs
        self.training_log = [] # train logs
        self.val_log = [] # val log


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
        self.training_log.append(loss.item())
        return {"loss": loss}

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        '''
            TODO: Add warm start
        '''
        # update params
        ## Add learning rate to tensorboard
        if self.logger:
            self.logger.experiment.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], self.trainer.global_step)
            
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()        
    
    # def on_batch_end(self):
    #     self.lr_log.append(self.optimizer.param_groups[0]["lr"])

    def validation_step(self, train_batch, batch_idx):
        images, labels = train_batch
        preds = self.forward(images)
        val_loss = self.loss_func(preds, labels)
        preds = preds.argmax(dim = 1)
        return {"y_pred": preds.cpu().numpy(),
            "y_true": labels.cpu().numpy(),
            "val_loss": val_loss}
    
    def validation_epoch_end(self, outputs):
        # tn, fp, fn, tp
        # Evaluate metrics
        y_pred, y_true = [], []
        for batch in outputs:
            y_pred = np.concatenate((y_pred, batch["y_pred"]))
            y_true = np.concatenate((y_true, batch["y_true"]))

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred) 
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        tensorboard_logs = {
            "accuracy_score": accuracy,
            "f1_score": f1,
            "recall_score": recall,
            "precision_score": precision
        }

        # self.logger.experiment.add_scalars("metrics", tensorboard_logs, self.current_epoch)
        # Add to one_cycle plot
        if (self.current_epoch + 1) % self.epoch_per_cycle == 0:
            self.logger.experiment.add_scalars("one_cycle", tensorboard_logs, self.current_cycle)
            '''
                TODO: add confusion matrix here
            '''

        # Get the total loss of the validation 
        total_loss = torch.stack([batch["val_loss"] for batch in outputs]).mean()
        self.val_log.append(total_loss.item())

        logs = {"val_loss": total_loss, "log": tensorboard_logs, "f1_score": f1}
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