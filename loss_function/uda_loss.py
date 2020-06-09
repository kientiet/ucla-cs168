'''
  Idea of UDA is from: https://arxiv.org/abs/1904.12848
  The code is referred from: https://github.com/google-research/uda/tree/960684e363251772a5938451d4d2bc0f1da9e24b

  Note: The code is translated and reduced by our understanding from the paper
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

KL_loss = nn.KLDivLoss(reduction = "none")
CrossEntropyLoss = nn.CrossEntropyLoss(reduction = "none")

def get_threshold(current_step, total_step, tsa_type, num_classes):
  if tsa_type == "log":
    alpha = 1 - math.exp(-current_step / total_step * 5.)
  elif tsa_type == "linear":
    alpha = current_step / total_step
  else:
    alpha = math.exp((current_step / total_step - 1) * 5.)

  return alpha * (1.0 - 1.0 / num_classes) + 1.0 / num_classes


def torch_device_one():
  return torch.tensor(1.).to(_get_device())


def _get_device():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return device


def anneal_sup_loss(supervised_logits,
                  supervised_labels,
                  supervised_loss,
                  total_step,
                  current_step,
                  num_classes,
                  tsa_type):

  prob = F.softmax(supervised_logits, dim = -1)
  one_hot_vector = F.one_hot(supervised_labels, num_classes)
  prob = torch.sum(prob * one_hot_vector, dim = -1)

  threshold = get_threshold(current_step, total_step, tsa_type, num_classes)
  loss_mask = 1. - (prob > threshold).type(torch.float32)

  supervised_loss = torch.sum(supervised_loss * loss_mask) / torch.max(torch.sum(loss_mask), torch_device_one())

  # threshold = get_threshold(current_step, total_step, tsa_type, num_classes)
  # larger_than_threshold = torch.exp(-supervised_loss) > threshold
  # loss_mask = torch.ones_like(supervised_labels, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
  # supervised_loss = torch.sum(supervised_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())

  return supervised_loss, threshold

def uda_loss(sup_images,
            labels,
            model,
            light_unsup_images = None,
            heavy_unsup_images = None,
            lambda_coeff = None,
            current_step = None,
            total_step = None,
            uda_confidence = -1,
            tempurature = -1,
            num_classes = 2,
            eval_mode = False,
            tsa_type = "exp"):

  supervised_logits = model(sup_images)
  supervised_loss = CrossEntropyLoss(supervised_logits, labels)

  if not eval_mode:
    # Eliminate the over-confident prediction
    supervised_loss, anneal_threshold = anneal_sup_loss(supervised_logits, labels, supervised_loss, total_step, current_step, num_classes, tsa_type)

    # Simple augmentation
    with torch.no_grad():
      light_unsup_logits = model(light_unsup_images)

      if tempurature != -1:
        ori_prob_temp = F.softmax(light_unsup_logits / tempurature, dim=-1)
      else:
        ori_prob_temp = F.softmax(light_unsup_logits, dim = -1)

      # Confidence masking
      if uda_confidence > 0:
        ori_prob = F.softmax(light_unsup_logits, dim = -1)
        largest_prob = torch.max(ori_prob, dim = -1)
        loss_mask = (largest_prob[0] > uda_confidence).type(torch.float32)
      else:
        loss_mask = torch.ones(light_unsup_logits.shape[0], dtype=torch.float32)

      loss_mask = loss_mask.to(_get_device())


    aug_logits = model(heavy_unsup_images)
    # Softmax temperature controlling
    # ! Note: The KL_loss(log_prob, prob) => aug_prob is used as log probability
    # if tempurature != -1:
    #   aug_logits = aug_logits / tempurature
    aug_log_prob = F.log_softmax(aug_logits, dim = -1)

    unsup_loss = torch.sum(KL_loss(aug_log_prob, ori_prob_temp.detach()), dim = -1)
    unsup_loss = torch.sum(unsup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())

    return supervised_loss + lambda_coeff * unsup_loss, supervised_loss, unsup_loss, anneal_threshold
  else:
    supervised_loss = torch.mean(supervised_loss)
    return supervised_loss, supervised_logits