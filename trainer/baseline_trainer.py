import torch
import torch.optim as optim

import numpy as np
from models.baseline import BaseLineModel
from trainer.trainer import TrainerSkeleton

class BaseLineTrainer(TrainerSkeleton):
    def __init__(self, trainloader, valloader, 
                num_classes = 2,
                base_lr = 1e-3,
                max_lr = 1e-2,
                num_cycle = 1,
                epoch_per_cycle = 4,
                running_scheduler = True
                ):
        super(BaseLineTrainer, self).__init__(trainloader = trainloader, valloader = valloader)

        # Load pretrained model
        self.model = BaseLineModel(num_classes)

        # Hyperparameters
        self.running_scheduler = True
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.num_cycle = num_cycle
        self.epoch_per_cycle = epoch_per_cycle
        self.current_cycle = 0

    
    def forward(self, inputs):
        return self.model(inputs)
        
    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.base_lr)

        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 
                            max_lr = self.max_lr, 
                            epochs = self.epoch_per_cycle,
                            steps_per_epoch = len(self.trainloader))
        return self.optimizer

    def on_epoch_end(self):
        if (self.current_epoch + 1) % self.epoch_per_cycle == 0:
            # This is only for 1 cycle
            self.current_cycle = (self.current_epoch + 1) // self.epoch_per_cycle
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                max_lr = self.max_lr, 
                                epochs = self.epoch_per_cycle,
                                steps_per_epoch = len(self.trainloader))

            avg_train = sum(self.training_log) / (self.epoch_per_cycle * len(self.trainloader))
            avg_val = sum(self.val_log) / self.epoch_per_cycle
            
            if self.logger is not None:
                self.logger.experiment.add_scalar("one_cycle/training_loss", avg_train, self.current_cycle)
                self.logger.experiment.add_scalar("one_cycle/val_loss", avg_val, self.current_cycle)
                
            self.training_log, self.val_log = [], []
    
    def get_max_epochs(self):
        return self.num_cycle * self.epoch_per_cycle