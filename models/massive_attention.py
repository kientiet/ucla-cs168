import torch
import torch.nn as nn
import torchvision.models as models

from models.transformers import Transformers
from models.senet import get_se_network

picture_dim = [512, 196, 49]

class MassiveAttention(nn.Module):
  def __init__(self,
              backbone,
              d_model = 512,
              dim_feedforward = 1024,
              num_classes = 2,
              reduce_backbone = None
              ):

    super().__init__()

    self.backbone = self.reduce_depth(backbone, reduce_backbone)
    self.d_model = d_model
    self.num_classes = num_classes

    # Define transformers
    backbone_channel = self.get_max_channel()
    self.transformers = Transformers(d_model = d_model, num_layers = 2, dim_feedforward = dim_feedforward)

    self.upsample = nn.Conv2d(backbone_channel, d_model, kernel_size = 1, bias = False)

    # Fully connected to output
    self.fc1 = nn.Linear(d_model, 1)

    ## It depends on the last
    reduce_backbone = (-1) if reduce_backbone is None else (-reduce_backbone - 1)
    self.fc2 = nn.Linear(picture_dim[reduce_backbone], num_classes - 1)


  def reduce_depth(self, backbone, layers):
    backbone = list(backbone.children())[:-2]
    if layers is not None:
      backbone = backbone[:-layers]

    return nn.Sequential(*backbone)

  def get_max_channel(self):
    max_channel = 0
    for p in self.backbone.modules():
      if isinstance(p, nn.Conv2d):
        max_channel = max(max_channel, p.out_channels)

    return max_channel

  def forward(self, inputs):
    '''
      # ! Note: This version will only use the global representation of the CNN for transformers
    '''
    features = self.backbone(inputs)

    if features.shape[1] != self.d_model:
      features = self.upsample(features)

    # Put to the transformers to get the final representations
    attention = self.transformers(features)

    logits = self.fc1(attention)
    logits = self.fc2(logits.squeeze())

    return logits

def resnet18_massiveattention(num_classes, reduce_backbone):
  return MassiveAttention(models.resnet18(pretrained = True), num_classes = num_classes, reduce_backbone = reduce_backbone)


def resnet34_massiveattention(num_classes, reduce_backbone):
  return MassiveAttention(models.resnet34(pretrained = True), num_classes = num_classes, reduce_backbone = reduce_backbone)


def resnet50_massiveattention(num_classes, reduce_backbone):
  return MassiveAttention(models.resnet50(pretrained = True), num_classes = num_classes, reduce_backbone = reduce_backbone)

def resnet18_se_massiveattention(num_classes, reduce_backbone):
  return MassiveAttention(get_se_network("resnet18_se"), num_classes = num_classes, reduce_backbone = reduce_backbone)

def resnet34_se_massiveattention(num_classes, reduce_backbone):
  return MassiveAttention(get_se_network("resnet34_se"), num_classes = num_classes, reduce_backbone = reduce_backbone)

def resnet50_se_massiveattention(num_classes, reduce_backbone):
  return MassiveAttention(get_se_network("resnet50_se"), num_classes = num_classes, reduce_backbone = reduce_backbone)

def resnext50_se_massiveattention(num_classes, reduce_backbone):
  return MassiveAttention(get_se_network("resnext50_32x4d_se"), num_classes = num_classes, reduce_backbone = reduce_backbone)


arch_list = {
  "resnet18": resnet18_massiveattention,
  "resnet34": resnet34_massiveattention,
  "resnet50": resnet50_massiveattention,
  "resnet18_se": resnet18_se_massiveattention,
  "resnet34_se": resnet34_se_massiveattention,
  "resnet50_se": resnet50_se_massiveattention,
  "resnext50_32x4d_se": resnext50_se_massiveattention,
}


def get_pretrained_net(netname, num_classes, reduce_backbone):
  return arch_list[netname](num_classes, reduce_backbone)