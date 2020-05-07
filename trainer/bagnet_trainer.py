import torch
import torch.nn
import torch.optim as optim

from models.bagnet import BagNetArchitecture
from trainer import TrainSkeleton

class BagNetTrainer(TrainSkeleton):
    def __init__(self, 
                trainloader, 
                valloader,
                num_classes = 2,
                base_lr = 1e-3,
                max_lr = 1e-2,
                num_cycle = 1,
                epoch_per_cycle = 4,
                running_scheduler = True
                ):

        super(BagNetTrainer, self).__init__(trainloader = trainloader, valloader = valloader)

        # Define the models
        self.model = BagNetArchitecture(in_channel = self.get_channel(), net_type = "bagnet7")

        # Define the hyperparameters
        self.running_scheduler = True
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.num_cycle = num_cycle
        self.epoch_per_cycle = epoch_per_cycle
    
    def forward(self, inputs):
        return self.model(inputs)

    '''
        TODO: Test the optimizer here
    '''
    def configure_optimizers(self):
        '''
            TODO: check if it works here
        '''
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.base_lr)

        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 
                            max_lr = self.max_lr, 
                            epochs = self.epoch_per_cycle,
                            steps_per_epoch = len(self.trainloader))
        return self.optimizer

    def on_epoch_end(self):
        if (self.current_epoch + 1) % self.epoch_per_cycle == 0:
            # This is only for 1 cycle
            print(">> Reset scheduler")
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                max_lr = self.max_lr, 
                                epochs = self.epoch_per_cycle,
                                steps_per_epoch = len(self.trainloader))