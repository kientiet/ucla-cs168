import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from loss_function.loss import get_loss_function
from models.backbone import get_back_bone
from evaluate.evaluator import Evaluator
from tqdm.notebook import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class SemiTrainer():
  def __init__(self,
              sup_loader,
              ori_loader,
              aug_loader,
              valloader,
              valset,
              model_name,
              lr_scheduler,
              momentum,
              save_model_dir,
              netname,
              run_on,
              base_lr = 0.0,
              max_lr = 0.0,
              warmup_step = 0,
              total_step = 0,
              epoch_per_cycle = 4,
              num_cycle = 1,
              loss_func = "uda",
              tensorboard_name = "uda",
              ):
    ## Assign dataloader
    self.sup_loader = sup_loader
    self.ori_loader = ori_loader
    self.aug_loader = aug_loader
    self.valloader = valloader
    # Evaluator
    self.evaluator = Evaluator(valset, netname = netname)

    ## Get the model and semi-loss function
    self.model = get_back_bone(model_name, pretrained = False)
    self.loss_func = get_loss_function(loss_func)

    ## Load hyperparameter hyperparameter
    self.save_model_dir = self.get_save_folder(save_model_dir)
    print(self.save_model_dir)
    self.netname = netname
    self.max_lr = max_lr
    self.base_lr = base_lr
    self.lr_scheduler = lr_scheduler
    self.epoch_per_cycle = epoch_per_cycle
    self.num_cycle = num_cycle
    self.total_step = total_step
    self.warmup_step = warmup_step

    self.optimizer = optim.SGD(self.model.parameters(), lr = self.base_lr, weight_decay = 5e-4, momentum = 0.9)
    self.reset_scheduler(lr_scheduler)

    ## Load tensorboard writer
    tensorboard_dir = self.get_tensorboard_dir(run_on, tensorboard_name)
    print(tensorboard_dir)
    self.writer = SummaryWriter(log_dir = tensorboard_dir)

  def train(self):
    total_steps = self.get_total_step()
    current_step, epoch = 0, -1
    ori_iter, aug_iter = iter(self.ori_loader), iter(self.aug_loader)
    print(self.optimizer)

    ## Load model to cuda
    self.model = self.model.to(device)
    while current_step <= total_steps:
      y_pred, y_true, y_pred_label = [], [], []
      epoch += 1

      for phase in ["train", "eval"]:
        print("\n>> Start to %s at %d epoch" % (phase, epoch + 1))
        if phase == "train":
          dataloader = self.sup_loader
        else:
          dataloader = self.valloader

        total_val_loss = 0.0
        for sup_images, labels in tqdm(dataloader, total = len(dataloader)):
          sup_images, labels = sup_images.to(device), labels.to(device)

          if phase == "train":
            current_step += 1

            self.model.train()
            try:
              ori_images = next(ori_iter)
              aug_images = next(aug_iter)
            except StopIteration:
              ori_iter, aug_iter = iter(self.ori_loader), iter(self.aug_loader)
              ori_images = next(ori_iter)
              aug_images = next(aug_iter)

            # Update the learning rate
            self.optimizer.zero_grad()
            self.update_lr(self.optimizer, current_step)

            ori_images, aug_images = ori_images.to(device), aug_images.to(device)

            semi_loss, supervised_loss, unsup_loss, anneal_threshold = self.calc_loss(sup_images, labels, eval_mode = False, current_step = current_step,
                                                                                      ori_images = ori_images, aug_images = aug_images)
            # Add loss to tensorboard
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], current_step)
            self.writer.add_scalar("anneal_threshold", anneal_threshold, current_step)
            self.writer.add_scalar("supervised_loss", supervised_loss.item(), current_step)
            self.writer.add_scalar("unsup_loss", unsup_loss.item(), current_step)

            # Update the semi_loss only and backprop
            semi_loss.backward()
            self.optimizer.step()

            ### ? Adding the clip weight here

          else:
            self.model.eval()
            with torch.no_grad():
              val_loss, logits = self.calc_loss(sup_images, labels, eval_mode = True)
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
        self.writer.add_scalar(key, value, epoch)

      print("\n>> Save model to %s" % self.save_model_dir)
      save_model_name = "{}_{}.pth".format(self.netname, epoch)
      self.save(save_model_name, epoch)

      ## Reset the cycle if necessary
      if (epoch + 1) % self.epoch_per_cycle == 0:
        self.reset_scheduler(self.lr_scheduler)

  def calc_loss(self, sup_images, labels, eval_mode, ori_images = None, aug_images = None, current_step = None):
    if eval_mode:
      val_loss, logits = self.loss_func(sup_images, labels, self.model, eval_mode = eval_mode)
      return val_loss, logits

    else:
      semi_loss, supervised_loss, unsup_loss, anneal_threshold = self.loss_func(sup_images, labels, self.model, ori_images, aug_images,
                    lambda_coeff = 1.0, current_step = current_step, total_step = self.get_total_step(),
                    uda_confidence = 0.8, tempurature = 0.4, tsa_type = "exp", eval_mode = eval_mode)

      return semi_loss, supervised_loss, unsup_loss, anneal_threshold

  def update_lr(self, optimizer, current_step):
    if current_step < self.warmup_step:
      new_lr = self.base_lr * current_step / self.warmup_step
      self.optimizer.param_groups[0]["lr"] = new_lr
    elif self.lr_scheduler == "cosine_anneling":
      coeff = (current_step - self.warmup_step) / (self.total_step - self.warmup_step)
      new_lr =  self.base_lr * np.cos(coeff * (7. * np.pi / 16.0))
      optimizer.param_groups[0]["lr"] = new_lr
    elif self.scheduler == "1cycle":
      self.scheduler.step()

  def get_total_epochs(self):
    return self.epoch_per_cycle * self.num_cycle

  def get_total_step(self):
    if self.total_step == 0:
      return self.epoch_per_cycle * self.num_cycle * len(self.sup_loader)
    else:
      return self.total_step

  def save(self, save_model_name, epoch):
    model_dir = os.path.join(self.save_model_dir, save_model_name)
    print(">> Saving model in %s" % model_dir)
    torch.save({
      "epoch": epoch,
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "scheduler_state_dict": self.scheduler.state_dict() if self.lr_scheduler == "1cycle" else None
    }, model_dir)

  def reset_scheduler(self, lr_scheduler):
    if lr_scheduler == "1cycle":
      self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr = self.max_lr,
                        epochs = self.epoch_per_cycle,
                        steps_per_epoch = len(self.sup_loader),
                        div_factor = 25,
                        final_div_factor = 1e4)

  def get_tensorboard_dir(self, run_on, tensorboard_name):
    save_dir = os.path.join(os.getcwd(), "logs", run_on, tensorboard_name)
    final_dir = self.get_save_folder(save_dir)
    return final_dir

  def get_save_folder(self, save_model_dir):
    version = "version_"
    i = 0
    while os.path.isdir(os.path.join(save_model_dir, version + str(i))):
      i += 1
    os.mkdir(os.path.join(save_model_dir, version + str(i)))
    return os.path.join(save_model_dir, version + str(i))