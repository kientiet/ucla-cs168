'''
  Transformers implemenetation is refered from
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertModel.forward
    and https://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#MultiheadAttention

  Note: Our Transformers only contain encoder
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Transformers(nn.Module):
  def __init__(self,
              d_model = 512,
              n_head = 8,
              num_layers = 2,
              dim_feedforward = 2048,
              position_encoding_style = "spatial",
              dropout = 0.1):

    super().__init__()
    self.position_encoding_style = position_encoding_style

    ## Get the positional encoding by channel as the features
    self.position_encoding = PositionEmbedding(d_model)

    ## Get a Transformer block
    self_attention_layer = TransformerLayer(d_model, n_head, dim_feedforward, dropout)
    ## Stack blocks
    self.encoder = TransformersBlock(self_attention_layer, num_layers)

    self.init_weight()

  def init_weight(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, inputs):
    batch_size, channels, h, w = inputs.shape

    # Get the positional encoding
    position_encoding = self.position_encoding(inputs)

    if self.position_encoding_style == "spatial":
      inputs = inputs.flatten(2).permute(2, 0, 1)
      position_encoding = position_encoding.flatten(2).permute(2, 0, 1)
    elif self.position_encoding_style == "channel":
      inputs = inputs.flatten(2).permute(1, 0, 2)
      position_encoding = position_encoding.flatten(2).permute(1, 0, 2)

    outputs = self.encoder(inputs, position_encoding)
    return outputs.permute(1, 0, 2)


class TransformersBlock(nn.Module):
  def __init__(self, layer, num_layers):
    super().__init__()

    self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
    self.num_layers = num_layers

  def forward(self, inputs, pos):
    outputs = inputs
    for layer in self.layers:
      outputs = layer(outputs, pos)

    return outputs


class TransformerLayer(nn.Module):
  def __init__(self,
              d_model,
              n_head,
              dim_forward = 2048,
              dropout_rate = 0.1
              ):

    super().__init__()
    self.self_attention = nn.MultiheadAttention(d_model, n_head, dropout_rate)

    self.linear1 = nn.Linear(d_model, dim_forward)
    self.linear2 = nn.Linear(dim_forward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout_rate)
    self.activation = nn.ReLU()

  def forward(self, inputs, pos):
    # Position encoding + inputs
    q = k = inputs + pos

    scores = self.self_attention(q, k, value = inputs)[0]
    inputs = inputs + self.dropout(scores)
    inputs = self.norm1(inputs)

    features = self.linear2(self.dropout(self.activation(self.linear1(inputs))))
    features = features + self.dropout(inputs)
    features = self.norm2(features)

    return features


class PositionEmbedding(nn.Module):
  def __init__(self, num_position = 256):
    super().__init__()
    self.num_position = num_position // 2
    self.row_embed = nn.Embedding(30, self.num_position)
    self.col_emded = nn.Embedding(30, self.num_position)

    self.init_weight()

  def init_weight(self):
    nn.init.uniform_(self.row_embed.weight)
    nn.init.uniform_(self.col_emded.weight)

  def forward(self, inputs):
    h, w = inputs.shape[-2:]
    i, j = torch.arange(w, device = inputs.device), torch.arange(h, device = inputs.device)
    x_emb, y_emb = self.col_emded(i), self.row_embed(j)

    pos = torch.cat([
      x_emb.unsqueeze(0).repeat(h, 1, 1),
      y_emb.unsqueeze(1).repeat(1, w, 1),
    ], dim = -1).permute(2, 0, 1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)

    return pos