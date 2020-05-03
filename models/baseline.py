import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class BaseLineModel(pl.LightningModule):
    def __init__(self, trainloader, valloader, 
                num_classes = 2,
                base_lr = 1e-3,
                max_lr = 1e-2,
                num_cycle = 1,
                epoch_per_cycle = 4,
                running_scheduler = True
                ):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_classes = num_classes

        # Load pretrained model
        self.resnet18 = torchvision.models.resnet18(pretrained = True)
        self.output_weight = nn.Linear(1000, num_classes)

        self.init_weight()

        # Define loss
        '''
            This will essentially be same as nn.BCELoss with nn.Sigmoid.
            Here, I will use CrossEntropyLoss in case later on, the num_classes
            is expanded then I don't have to change the loss function
            TODO: Explore different cross-entropy
        '''
        self.criterion = nn.CrossEntropyLoss()

        # Hyperparameters
        self.running_scheduler = True
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.num_cycle = num_cycle
        self.epoch_per_cycle = epoch_per_cycle

        # learning rate logs
        self.logs = []
    
    def init_weight(self):
        nn.init.normal_(self.output_weight.weight.data, mean = 0, std = 0.01)

    def forward(self, inputs):
        inputs = self.resnet18(inputs)
        outputs = self.output_weight(inputs)
        return outputs
        
    def loss_func(self, logits, labels):
        return self.criterion(logits, labels)

    def freeze(self, num_layers = 1, is_testing = False):
        '''
            TODO: How to freeze layers of ResNet?
        '''
        total_layers = len(list(self.children()))
        if num_layers is None: num_layers = total_layers + 1

        for index, layer in enumerate(self.children()):
            for param in layer.parameters():
                if index < total_layers - num_layers:
                    param.requires_grad = False
                else: 
                    param.requires_grad = True

        if is_testing:
            for name, p in self.named_parameters():
                print(name, p.requires_grad)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()), lr = self.base_lr)

        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 
                            max_lr = self.max_lr, 
                            epochs = self.epoch_per_cycle,
                            steps_per_epoch = len(self.trainloader))
        return [self.optimizer]#, [self.scheduler]
    
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
    
    def on_epoch_end(self):
        if (self.current_epoch + 1) % self.epoch_per_cycle == 0:
            # This is only for 1 cycle
            print(">> Reset scheduler")
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                max_lr = self.max_lr, 
                                epochs = self.epoch_per_cycle,
                                steps_per_epoch = len(self.trainloader))

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
    
    def show_lr(self):
        plt.plot(range(len(self.logs)), self.logs)
        plt.show()

'''
    TODO: 1/ how to save as checkpoint
        2/ How to reload and run check point
'''