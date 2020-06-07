import torch
import numpy as np
import copy

class ExponentialMovingAverage:
  def __init__(self, model, device, decay = 0.999, bias_correction = True, num_updates = False):
    self.decay = decay
    self.bias_correction = bias_correction
    self.device = device
    self.model = self.clone(model, init = True)
    self.num_updates = 0. if num_updates else None


  def update(self, model):
    decay = self.decay
    if self.num_updates is not None:
      decay = min(decay, (1. + self.num_updates) / (10. + self.num_updates))
      self.num_updates += 1

    parameters = self.clone(model)
    for name, new_param in parameters.items():
      curr_param = self.model[name]
      curr_param.sub_((1 - decay) * (curr_param - new_param))
      self.model[name] = curr_param


  def clone(self, model, init = False):
    clone_model = copy.deepcopy(model)
    clone_model = clone_model.cpu()

    state_dict = {}
    for name, p in clone_model.named_parameters():
      if p.requires_grad:
        state_dict[name] = p.clone().detach()

    return state_dict

  def load_weight(self, model):
    print("\n\n>> Load Exponential Average Weight...")
    new_model = copy.deepcopy(model)
    for name, p in new_model.named_parameters():
      if name in self.model:
        p.data = self.model[name]

    return new_model.to(self.device)