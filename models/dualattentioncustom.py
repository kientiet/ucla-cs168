import torch
import torch.nn as nn

from models.dualattention import PointAttentionBlock, ChannelAttentionBlock

class DualAttentionCustom(nn.Module):
  def __init__(self,
              channel_layers,
              input_dim):
    ## Query is the global block
    ## Key is the local features
    super().__init__()
    self._calibrate_dimension(channel_layers, input_dim)

    # Both PointAttention and ChannelAttention will be shared
    self.point_attention = PointAttentionBlock(channel_layers[-1])
    self.channel_attention = ChannelAttentionBlock(channel_layers[-1])

    # Fused operation among
    self.point_fused = self._make_fused_layer(channel_layers, "position_attention_")
    self.channel_fused = self._make_fused_layer(channel_layers, "channel_attention_")

    # Output channel
    self.outputs = self._make_output(channel_layers)

  def _make_fused_layer(self, channel_layers, prefix_name):
    fused = nn.ModuleList([])
    global_channel = channel_layers[-1]
    for _ in channel_layers:
      fused.append(nn.Sequential(
        nn.Conv2d(global_channel, global_channel, kernel_size = 3, padding = 1, bias = False),
        nn.BatchNorm2d(global_channel),
        nn.ReLU(),
        nn.Dropout2d(0.1, False),
        nn.Conv2d(global_channel, global_channel, kernel_size = 1)
      ))

    return fused

  def _make_output(self, channel_layers):
    outputs = nn.ModuleList([])
    global_channel = channel_layers[-1]
    for _ in channel_layers:
      outputs.append(nn.Sequential(
        nn.Dropout2d(0.1, False),
        nn.Conv2d(global_channel, global_channel, kernel_size = 1)
      ))

    return outputs

  def _calibrate_dimension(self, channel_layers, input_dim):
    global_dim, global_channel = input_dim[-1], channel_layers[-1]
    self.down_sample = nn.ModuleList([])
    for index, (dim, channel) in enumerate(zip(input_dim, channel_layers)):
      self.down_sample.append(
        nn.Conv2d(channel, global_channel, kernel_size = 3, stride = (dim - 1) // (global_dim - 1),
                padding = 1, bias = False)
      )

  def forward(self, local_features, global_features):
    ## Add the global_features to local_features to get the representation
    local_features = local_features +  [global_features.clone()]

    batch, channel, _, _ = local_features[-1].shape
    ## This will return a vector HxW - dimension
    features = []
    for index, local in enumerate(local_features):
      # Stem the local features
      inputs_fused = self.down_sample[index](local)

      # ? Should fuse the global features

      # Get point attention
      point_attention = self.point_attention(inputs_fused, global_features)
      point_fused = self.point_fused(point_attention)

      # Get channel attention
      channel_attention = self.channel_attention(inputs_fused, global_features)
      channel_fused = self.channel_fused(channel_attention)

      # Get the fusion of cross channel attention
      sum_fused = point_fused + channel_fused

      ### To avoid exploit due to the sum, we use the sigmoid to bound the value here
      outputs = self.outputs(sum_fused)

      # Transform to batch x channel
      outputs = outputs.view(batch, channel, -1).sum(-1)
      outputs = torch.sigmoid(outputs.squeeze())

      features.append(outputs)

    features = torch.stack(features, dim = 1)

    return features

class DualAttentionCustomShared(nn.Module):
  def __init__(self,
              channel_layers,
              input_dim):
    ## Query is the global block
    ## Key is the local features
    super().__init__()
    self._calibrate_dimension(channel_layers, input_dim)

    # Both PointAttention and ChannelAttention will be shared
    self.point_attention = PointAttentionBlock(channel_layers[-1])
    self.channel_attention = ChannelAttentionBlock(channel_layers[-1])

    # Fused operation among
    self.point_fused = self._make_fused_layer(channel_layers, "position_attention_")
    self.channel_fused = self._make_fused_layer(channel_layers, "channel_attention_")

    # Output channel
    self.outputs = self._make_output(channel_layers)

    self.init_weight()

  def init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()

  def _make_fused_layer(self, channel_layers, prefix_name):
    global_channel = channel_layers[-1]
    fused = nn.Sequential(
      nn.Conv2d(global_channel, global_channel, kernel_size = 3, padding = 1, bias = False),
      nn.BatchNorm2d(global_channel),
      nn.ReLU(),
      nn.Dropout2d(0.1, False),
      nn.Conv2d(global_channel, global_channel, kernel_size = 1)
    )
    return fused

  def _make_output(self, channel_layers):
    global_channel = channel_layers[-1]
    outputs = nn.Sequential(
      nn.Dropout2d(0.2),
      nn.Conv2d(global_channel, global_channel, kernel_size = 1)
    )

    return outputs

  def _calibrate_dimension(self, channel_layers, input_dim):
    global_dim, global_channel = input_dim[-1], channel_layers[-1]
    self.down_sample = nn.ModuleList([])
    for index, (dim, channel) in enumerate(zip(input_dim, channel_layers)):
      self.down_sample.append(
        nn.Conv2d(channel, global_channel, kernel_size = 3, stride = (dim - 1) // (global_dim - 1),
                padding = 1, bias = False)
      )

  def forward(self, local_features, global_features):
    ## Add the global_features to local_features to get the representation
    local_features = local_features +  [global_features.clone()]

    batch, channel, _, _ = local_features[-1].shape
    ## This will return a vector HxW - dimension
    features = []
    for index, local in enumerate(local_features):
      # Stem the local features
      inputs_fused = self.down_sample[index](local)

      # Get point attention
      point_attention = self.point_attention(inputs_fused, global_features)
      point_fused = self.point_fused(point_attention)

      # Get channel attention
      channel_attention = self.channel_attention(inputs_fused, global_features)
      channel_fused = self.channel_fused(channel_attention)

      # Get the fusion of cross channel attention
      sum_fused = point_fused + channel_fused

      ### To avoid exploit due to the sum, we use the sigmoid to bound the value here
      outputs = self.outputs(sum_fused)

      # Transform to batch x channel
      outputs = outputs.view(batch, channel, -1).sum(-1)
      outputs = torch.sigmoid(outputs.squeeze())

      features.append(outputs)

    features = torch.stack(features, dim = 1)

    return features
