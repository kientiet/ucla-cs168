import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class BaseLineModel(nn.Module):
    def __init__(self, model_type = "resnet18", num_classes = 2, backbone = None, keep = False):
        super(BaseLineModel, self).__init__()
        self.num_classes = num_classes

        # Load pretrained model
        self.resnet = backbone if backbone is not None else torchvision.models.resnet18(pretrained = True)
        if keep:
            self.output_weight = nn.Linear(1000, num_classes)
        else:
            self.output_weight = nn.Linear(1000, num_classes - 1)

        self.init_weight()

    def forward(self, inputs):
        inputs = self.resnet(inputs)
        outputs = self.output_weight(inputs)
        return outputs

    def init_weight(self):
        nn.init.normal_(self.output_weight.weight.data, mean = 0, std = 0.01)