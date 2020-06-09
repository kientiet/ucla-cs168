'''
  The idea of SENet is from: https://arxiv.org/abs/1709.01507

  The code is referenced from: https://github.com/moskomule/senet.pytorch
    We did make adjustment to work with our setup
'''

import torch.nn as nn
import math
from torchvision.models.resnet import ResNet, _resnet
from torchvision.models.utils import load_state_dict_from_url

def conv3x3(in_planes, out_planes, stride = 1):
  return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)


def conv1x1(in_planes, out_planes, stride = 1):
  return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)


class SELayer(nn.Module):
  def __init__(self, channel, reduction = 16):
    super().__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias = False),
      nn.ReLU(inplace = True),
      nn.Linear(channel // reduction, channel, bias = False),
      nn.Sigmoid()
    )

    self.init_weight(channel)

  def init_weight(self, channel):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        total_dim = m
        nn.init.normal_(m.weight, 0, math.sqrt(2. / pow(channel, 2)))

  def forward(self, x):
    batch, channel, _, _ = x.shape
    out = self.avg_pool(x).view(batch, channel)
    out = self.fc(out).view(batch, channel, 1, 1)

    return x * out.expand_as(x)


class SEBasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None, *, reduction = 16):
    super(SEBasicBlock, self).__init__()

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace = True)

    self.conv2 = conv3x3(planes, planes, 1)
    self.bn2 = nn.BatchNorm2d(planes)

    self.se = SELayer(planes, reduction)

    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(residual)

    out += residual
    out = self.relu(out)

    return out

class SEBottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None, *, reduction = 16):
    super(SEBottleneck, self).__init__()
    self.conv1 = conv1x1(inplanes, planes)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = conv3x3(planes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)

    self.conv3 = conv1x1(planes, planes * 4)
    self.bn3 = nn.BatchNorm2d(planes * 4)

    self.relu = nn.ReLU(inplace = True)

    self.se = SELayer(planes * 4, reduction)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
}

def resnet18_se():
  model = ResNet(SEBasicBlock, [2, 2, 2, 2])

  state_dict = load_state_dict_from_url(model_urls["resnet18"], progress = True)
  model.load_state_dict(state_dict, strict = False)

  return model

def resnet34_se():
  model = ResNet(SEBasicBlock, [3, 4, 6, 3])

  state_dict = load_state_dict_from_url(model_urls["resnet34"], progress = True)
  model.load_state_dict(state_dict, strict = False)

  return model

def resnet50_se():
  model = ResNet(SEBottleneck, [3, 4, 6, 3])

  state_dict = load_state_dict_from_url(model_urls["resnet50"], progress = True)
  model.load_state_dict(state_dict, strict = False)

  return model

def resnext50_32x4d_se(**kwargs):
  kwargs["groups"] = 32
  kwargs["width_per_group"] = 4

  model = ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
  state_dict = load_state_dict_from_url(model_urls["resnext50_32x4d"], progress = True)
  model.load_state_dict(state_dict, strict = False)

  return model


list_network = {
  "resnet18_se": resnet18_se,
  "resnet34_se": resnet34_se,
  "resnet50_se": resnet50_se,
  "resnext50_32x4d_se": resnext50_32x4d_se
}

def get_se_network(netname):
  return list_network[netname]()