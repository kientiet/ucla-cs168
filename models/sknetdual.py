import torch
import torch.nn as nn
from models.sknet import sknet18_32x4d
from models.dualattention import DualAttentionHead

class SKNetDual(nn.Module):
  def __init__(self, num_classes = 2):
    super().__init__()
    self.sknet = sknet18_32x4d(num_classes)
    self.sknet = nn.Sequential(*(list(self.sknet.children())[:-1]))

    in_channel = 2048
    self.dual_attention = DualAttentionHead(in_channel, in_channel)

    # From fc to correct outputs
    self.num_classes = num_classes
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(0.2)
    self.fc = nn.Linear(in_channel, num_classes -1)

  def forward(self, inputs):
    features = self.sknet(inputs)
    outputs = self.dual_attention(features)
    outputs = self.avg_pool(outputs)
    outputs = outputs.view(inputs.shape[0], -1)
    outputs = self.fc(outputs)

    return outputs
