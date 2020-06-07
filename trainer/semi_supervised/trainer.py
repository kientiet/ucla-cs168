import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from evaluate.evaluator import Evaluator
from trainer.semi_supervised.exp_moving_average import ExponentialMovingAverage
from models.backbone import get_back_bone
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class TrainerSkeleton:
  def __init__(self,
              sup_loader,
              ori_loader,
              aug_loader,
              valloader,
              valset,
              json_config
              ):

    super().__init__()
    self.config = json_config

    # Assign dataset
    self.sup_loader = sup_loader
    self.ori_loader = ori_loader
    self.aug_loader = aug_loader
    self.valloader = valloader

    # Get training config
    self.model = get_back_bone(self.config["backbone"], pretrained = self.config["pretrained"])

    # Evaluator engine
    self.netname = self.config["netname"]
    self.evaluator = Evaluator(valset, netname = self.config["netname"])

    # Hyperparameters
    self.max_lr = self.config["max_lr"]
    self.base_lr = self.config["base_lr"]
    self.lr_scheduler = self.config["lr_scheduler"]
    self.epoch_per_cycle = self.config["epoch_per_cycle"]
    self.num_cycle = self.config["num_cycle"]
    self.total_epoch = self.config["total_epoch"]
    self.total_step = self.config["total_step"]
    self.warmup_step = self.config["warmup_step"]

    # Save and load file
    if self.config["running_mode"] == "debug":
      self.writer = None
    else:
      self.save_model_dir = self.get_save_folder(self.config["save_model_dir"])
      print(self.save_model_dir)
      tensorboard_dir = self.get_tensorboard_dir(self.config["run_on"], self.config["tensorboard_name"])
      print(tensorboard_dir)
      self.writer = SummaryWriter(log_dir = tensorboard_dir)

    # Apply EMA
    self.ema = ExponentialMovingAverage(self.model, device, num_updates = self.config["num_updates"])

  def train(self):
    self.current_step, self.epoch = 0, -1
    ori_iter, aug_iter = iter(self.ori_loader), iter(self.aug_loader)
    print(self.optimizer)

    ## Load model to cuda
    self.model = self.model.to(device)
    while self.stop_condition():
      y_pred, y_true, y_pred_label = [], [], []
      self.epoch += 1

      for phase in ["train", "eval"]:
        print("\n>> Start to %s at %d epoch" % (phase, self.epoch + 1))
        if phase == "train":
          dataloader = self.sup_loader
          self.model.train()
        else:
          self.model.eval()
          new_model = self.ema.load_weight(self.model)
          dataloader = self.valloader

        total_val_loss = 0.0
        for sup_images, labels in tqdm(dataloader, total = len(dataloader)):
          sup_images, labels = sup_images.to(device), labels.to(device)

          if phase == "train":
            self.current_step += 1

            try:
              ori_images = next(ori_iter)
              aug_images = next(aug_iter)
            except StopIteration:
              ori_iter, aug_iter = iter(self.ori_loader), iter(self.aug_loader)
              ori_images = next(ori_iter)
              aug_images = next(aug_iter)

            # Update the learning rate
            self.optimizer.zero_grad()
            self.update_lr(self.optimizer, self.current_step)

            ori_images, aug_images = ori_images.to(device), aug_images.to(device)

            semi_loss, sup_loss, unsup_loss = self.train_loss(sup_images, labels, ori_images, aug_images)

            # Update the semi_loss only and backprop
            semi_loss.backward()
            self.optimizer.step()

            # EMA update
            self.ema.update(self.model)

          else:
            with torch.no_grad():
              val_loss, logits = self.eval_loss(sup_images, labels)
              total_val_loss = np.append(total_val_loss, val_loss.item())

              # Get the prediction for label 1
              pred = F.softmax(logits, dim = -1)
              prob = pred[:, 1]
              _, pred_label = torch.max(pred, dim = -1)

              y_pred = np.concatenate((y_pred, prob.cpu().numpy()))
              y_pred_label = np.concatenate((y_pred_label, pred_label.cpu().numpy()))
              y_true = np.concatenate((y_true, labels.cpu().numpy()))

      print("\n>> Run evaluation...")

      tensorboard_logs = self.evaluator.eval_on_test_set(y_true, y_pred, y_pred_label)
      tensorboard_logs["val_loss"] = np.mean(total_val_loss)
      for key, value in tensorboard_logs.items():
        self.writer.add_scalar(key, value, self.epoch)

      print("\n>> Save model to %s" % self.save_model_dir)
      save_model_name = "{}_{}.pth".format(self.netname, self.epoch)
      self.save(save_model_name, self.epoch)

      ## Reset the cycle if necessary
      if (self.epoch_per_cycle != 0) and ((self.epoch + 1) % self.epoch_per_cycle == 0):
        self.reset_scheduler(self.lr_scheduler)

  def train_loss(self):
    pass

  def eval_loss(self):
    pass

  def get_tensorboard_dir(self, run_on, tensorboard_name):
    save_dir = os.path.join(os.getcwd(), "logs", run_on, tensorboard_name)
    final_dir = self.get_save_folder(save_dir)
    return final_dir

  def get_save_folder(self, save_model_dir):
    save_model_dir = os.path.join(os.getcwd(), save_model_dir)
    version = "version_"
    i = 0
    while os.path.isdir(os.path.join(save_model_dir, version + str(i))):
      i += 1
    os.mkdir(os.path.join(save_model_dir, version + str(i)))
    return os.path.join(save_model_dir, version + str(i))

  def stop_condition(self):
    if self.get_total_epochs() > 0: return self.epoch <= self.get_total_epochs()
    return self.current_step <= self.total_step

  def get_total_epochs(self):
    if self.total_epoch > 0:
      return self.total_epoch
    elif self.num_cycle > 0:
      return self.epoch_per_cycle * self.num_cycle
    else:
      return 0

  def get_total_step(self):
    return self.total_step

  def reset_scheduler(self, lr_scheduler):
    if lr_scheduler == "1cycle":
      self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr = self.max_lr,
                        epochs = self.epoch_per_cycle,
                        steps_per_epoch = len(self.sup_loader),
                        div_factor = 25,
                        final_div_factor = 1e4)

  def update_lr(self, optimizer, current_step):
    if current_step < self.warmup_step:
      new_lr = self.max_lr * current_step / self.warmup_step
      self.optimizer.param_groups[0]["lr"] = new_lr
    elif self.lr_scheduler == "cosine_anneling":
      coeff = (current_step - self.warmup_step) / (self.total_step - self.warmup_step)
      new_lr =  self.max_lr * np.cos(coeff * (7. * np.pi / 16.0))
      optimizer.param_groups[0]["lr"] = new_lr
    elif self.lr_scheduler == "linear":
      for threshold in self.lr_threshold:
        if self.epoch == threshold:
          new_lr = optimizer.param_groups[0]["lr"] * 0.1
          break

  def save(self, save_model_name, epoch):
    model_dir = os.path.join(self.save_model_dir, save_model_name)
    print(">> Saving model in %s" % model_dir)
    torch.save({
      "epoch": epoch,
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "scheduler_state_dict": self.scheduler.state_dict() if self.lr_scheduler == "1cycle" else None
    }, model_dir)