import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.baseline import BaseLineModel

from sklearn.metrics import confusion_matrix
import numpy as np

class BaseLineTrainer(pl.LightningModule):
    def __init__(self, trainloader, valloader):
        super().__init__()
        self.model = BaseLineModel(trainloader, valloader)
        self.trainloader = trainloader
        self.valloader = valloader
    
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)

        # Define loss
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, inputs):
        return self.model(inputs)
    
    def loss_func(self, logits, labels):
        return self.criterion(F.softmax(logits, dim = 1), labels)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        images, labels = images.to(self.device), labels.to(self.device)
        logits = self.model(images)
        loss = self.loss_func(logits, labels)

        return {"loss": loss}

    def validation_step(self, train_batch, batch_idx):
        images, labels = train_batch
        images, labels = images.to(self.device), labels.to(self.device)
        with torch.no_grad():
            preds = F.softmax(self.model(images), dim = 1)
            preds = torch.argmax(preds, dim = 1)

        return {"confusion_matrix": confusion_matrix(y_true = labels.cpu().numpy(), y_pred = preds.cpu().numpy())}
    
    def validation_epoch_end(self, outputs):        
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