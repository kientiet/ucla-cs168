import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

class BaseLineModel(nn.Module):
    def __init__(self, num_classes = 2):
        super(BaseLineModel, self).__init__()
        self.num_classes = num_classes

        # Load pretrained model
        self.resnet18 = torchvision.models.resnet18(pretrained = True)
        self.output_weight = nn.Linear(1000, num_classes)

        self.init_weight()
    
    def forward(self, inputs):
        inputs = self.resnet18(inputs)
        outputs = self.output_weight(inputs)
        return outputs
        
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

    def init_weight(self):
        nn.init.normal_(self.output_weight.weight.data, mean = 0, std = 0.01)