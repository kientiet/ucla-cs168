import os
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from trainer.semi_supervised.trainer import TrainerSkeleton
from loss_function.fixmatch_loss import fixmatch_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
json_config_dir = "trainer/semi_supervised/fixmatch_config.yml"

class FixMatchTrainer(TrainerSkeleton):
  def __init__(self,
              sup_loader,
              ori_loader,
              aug_loader,
              valloader,
              valset
              ):

    with open(json_config_dir, "r") as f:
      config = yaml.load(f, Loader = yaml.FullLoader)
    print(config)

    super(FixMatchTrainer, self).__init__(sup_loader, ori_loader, aug_loader, valloader, valset, config)

    self.optimizer = optim.SGD(self.model.parameters(), lr = config["base_lr"], weight_decay = config["weight_decay"], momentum = config["momentum"], nesterov = config["nesterov"])
    if self.warmup_step > 0: self.update_lr(self.update_lr, 1)
    self.reset_scheduler(config["lr_scheduler"])

    # Training related
    self.lambda_coeff = config["lambda_coeff"]
    self.fixmatch_confidence = config["fixmatch_confidence"]

  def train_loss(self, sup_images, labels, ori_images, aug_images):
    semi_loss, supervised_loss, unsup_loss = fixmatch_loss(self.model, sup_images, labels, ori_images, aug_images,
                                                          self.lambda_coeff, self.fixmatch_confidence)

    # Add loss to tensorboard
    self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], self.current_step)
    self.writer.add_scalar("supervised_loss", supervised_loss.item(), self.current_step)
    self.writer.add_scalar("unsup_loss", unsup_loss.item(), self.current_step)

    return semi_loss, supervised_loss, unsup_loss


  def eval_loss(self, sup_images, labels):
    supervised_loss, supervised_logits = fixmatch_loss(self.model, sup_images, labels, eval_mode = True)
    return supervised_loss, supervised_logits