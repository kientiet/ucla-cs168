import torch
import torch.nn as nn

from models.dualattentioncustom import DualAttentionCustom, DualAttentionCustomShared
from models.selfattention import SelfAttentionHead
from models.sknet import SKBlock

class MassiveAttention(nn.Module):
  def __init__(self,
              block,
              layers,
              cardinality = None,
              bottleneck_width = None,
              r = None,
              expansion = 2,
              out_dim = 2048,
              extract_layers = 2,
              num_classes = 2,
              shared_weight = True
              ):

    super().__init__()

    # Define necessary elements for Net
    self.in_planes = 64
    self.cardinality = cardinality
    self.bottleneck_width = bottleneck_width
    self.r = r
    self.expansion = expansion
    self.extract_layers = extract_layers - 1

    # Stem head
    self.stem_head = nn.Sequential(
      nn.Conv2d(in_channels = 3, out_channels = self.in_planes, kernel_size = 7, stride = 2, padding = 2),
      nn.BatchNorm2d(self.in_planes),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    )

    # Make the layers for net
    self.channels, self.dim = [], [56, 28, 14, 7]
    self.layer_1 = self._make_layers(block, layers[0], 1)
    self.layer_2 = self._make_layers(block, layers[1], 2)
    self.layer_3 = self._make_layers(block, layers[2], 2)
    self.layer_4 = self._make_layers(block, layers[3], 2)

    # Dual Attention Head
    if shared_weight:
      self.dualattention = DualAttentionCustomShared(self.channels[self.extract_layers:], self.dim[self.extract_layers:])
    else:
      self.dualattention = DualAttentionCustom(self.channels[self.extract_layers:], self.dim[self.extract_layers:])

    # Transformers head
    self.transformer = SelfAttentionHead(self.channels[-1])

    # Mapping to output
    self.outputs = nn.Linear(self.transformer.attention_heads.project_dim, num_classes - 1)

  def _make_layers(self, block, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(
        block(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion, self.r)
      )
      self.in_planes = self.expansion * self.bottleneck_width * self.cardinality
    self.channels.append(self.in_planes)

    self.bottleneck_width *= 2
    return nn.Sequential(*layers)

  def forward(self, inputs):
    features = self.stem_head(inputs)
    store = []
    features = self.layer_1(features)
    store.append(features.clone())

    features = self.layer_2(features)
    store.append(features.clone())

    features = self.layer_3(features)
    store.append(features.clone())

    features = self.layer_4(features)
    global_features = features.clone()

    # This will return modulelist of extract_layers
    attention = self.dualattention(store[self.extract_layers:], global_features)

    # Put to the transformers to get the final representations
    attention = self.transformer(attention)

    # ** Only get the last vector as the context vector
    outputs = self.outputs(attention[:,-1, :])
    return outputs

def massive_attention_16x4d(num_classes):
  return MassiveAttention(block = SKBlock,
                        layers = [2, 2, 2, 2], cardinality = 32, bottleneck_width = 4, r = 16, num_classes = num_classes)