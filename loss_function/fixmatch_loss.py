'''
  FixMatch paper can be found here: https://arxiv.org/abs/2001.07685
  The code is referred from: https://github.com/google-research/fixmatch

  Note: The code is translated and reduced by our understanding from the paper
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

unsup_loss_func = nn.CrossEntropyLoss(reduction = "none")
sup_loss_func = nn.CrossEntropyLoss()

def torch_device_one():
  return torch.tensor(1.).to(_get_device())


def _get_device():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return device


def fixmatch_loss(model,
                  sup_images,
                  labels,
                  light_unsup_images = None,
                  heavy_unsup_images = None,
                  lambda_coeff = None,
                  fixmatch_confidence = -1,
                  eval_mode = False,
            ):

  # Supervised loss
  supervised_logits = model(sup_images)
  supervised_loss = sup_loss_func(input = supervised_logits, target = labels)

  if not eval_mode:
    # Simple augmentation
    with torch.no_grad():
      light_unsup_logits = model(light_unsup_images)

      # Confidence masking
      if fixmatch_confidence > 0:
        pseudo_labels = F.softmax(light_unsup_logits, dim = -1)
        largest_prob = torch.max(pseudo_labels, dim = -1)
        loss_mask = (largest_prob[0] >= fixmatch_confidence).type(torch.float32)
      else:
        loss_mask = torch.ones(light_unsup_logits.shape[0], dtype=torch.float32)

      loss_mask = loss_mask.to(_get_device())


    aug_logits = model(heavy_unsup_images)
    unsup_loss = unsup_loss_func(input = aug_logits, target = torch.argmax(pseudo_labels.detach(), dim = -1).detach())
    unsup_loss = torch.sum(unsup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())

    return supervised_loss + lambda_coeff * unsup_loss, supervised_loss, unsup_loss
  else:
    supervised_loss = torch.mean(supervised_loss)
    return supervised_loss, supervised_logits