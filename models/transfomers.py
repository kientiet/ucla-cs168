import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionBlock(nn.Module):
  def __init__(self,
              num_head,
              hidden_dim = 2048,
              num_blocks = 4,
              dropout_rate = 0.1
              ):

    super().__init__()
    assert hidden_dim % num_blocks == 0

    self.num_blocks = num_blocks
    self.hidden_dim = hidden_dim
    self.project_dim = hidden_dim // 4

    self.num_attention_heads = num_head
    self.attention_head_size = int(self.project_dim / num_head)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.project_query = nn.Linear(hidden_dim, self.all_head_size)
    self.project_value = nn.Linear(hidden_dim, self.all_head_size)
    self.project_key = nn.Linear(hidden_dim, self.all_head_size)

    self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

  def forward(self, hidden_states):
    # TODO: Check the dimension here to avoid exploit the memory
    query = self.project_query(hidden_states)
    value = self.project_value(hidden_states)
    key = self.project_value(hidden_states)

    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
    probs = F.softmax(scores, dim = -1)

    if self.dropout is not None:
      probs = self.dropout(probs)

    outputs = torch.matmul(probs, value)

    return outputs

class FeedForwardLayer(nn.Module):
  def __init__(self,
              hidden_dim,
              dropout_rate = 0.1
              ):

    super().__init__()
    self.hidden_dim = hidden_dim
    self.dense = nn.Linear(hidden_dim, hidden_dim)
    self.layer_norm = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

  def forward(self, hidden_states):
    residual = hidden_states
    features = self.dense(hidden_states)
    features = self.dropout(features)
    outputs = self.layer_norm(features + residual)

    return outputs

class SelfAttentionHead(nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.attention_heads = SelfAttentionBlock(num_head = 4)
    self.fc = FeedForwardLayer(self.attention_heads.project_dim)

  def forward(self, features):
    attention = self.attention_heads(features)
    outputs = self.fc(attention)

    return outputs