import torch
import torch.optim as optim
from models.bagnet import BagNetArchitecture
from trainer.trainer import TrainerSkeleton

class BagNetTrainer(TrainerSkeleton):
    def __init__(self, trainloader, valloader, 
                num_classes = 2,
                base_lr = 1e-3,
                max_lr = 1e-2,
                num_cycle = 1,
                epoch_per_cycle = 4,
                running_scheduler = True
                ):
        super(BagNetTrainer, self).__init__(trainloader = trainloader, valloader = valloader)

        # Load pretrained model
        self.model = BagNetArchitecture(in_channel = 3, num_classes = num_classes)

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
        '''
            TODO: check if it works here
                Should we add weight_decay?
        '''
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.base_lr)

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

            self.logger.experiment.add_scalar("one_cycle/training_loss", avg_train, self.current_cycle)
            self.logger.experiment.add_scalar("one_cycle/val_loss", avg_val, self.current_cycle)
            self.training_log, self.val_log = [], []
    
    def get_max_epochs(self):
        return self.num_cycle * self.epoch_per_cycle