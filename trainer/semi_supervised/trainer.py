'''
	# TODO: 2 class with cross-entropy instead of BCELoss
'''

import pytorch_lightning as pl
import torch

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import numpy as np

from loss_function.uda_loss import uda_loss

class SemiTrainerSkeleton(pl.LightningModule):
	def __init__(self,
							trainloader,
							valloader,
							criterion_func
							):
		super(SemiTrainerSkeleton, self).__init__()
		self.trainloader = trainloader
		self.valloader = valloader

		# Define loss
		self.criterion = criterion_func

		# Logging config
		self.training_log = [] # train logs
		self.val_log = [] # val log

		# Hyperparameters
		self.threshold = 0.5

	def forward(self):
		pass

	def configure_optimizers(self):
		pass

	def total_epoch(self):
		return self.epoch_per_cycle * self.num_cycle

	def training_step(self, train_batch, batch_idx):
		images, labels = train_batch
		loss, _ = self.criterion(images, labels, self.model)
		self.training_log.append(loss.item())
		return {"loss": loss}

	def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
		# update params
		## Add learning rate to tensorboard
		if self.logger:
			self.logger.experiment.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], self.trainer.global_step)

		self.optimizer.step()
		self.scheduler.step()
		self.optimizer.zero_grad()

	def validation_step(self, batch, batch_idx):
		images, labels = batch
		val_loss, logits = self.loss_func(images, labels, self.model)
		return {"y_pred": logits.cpu().numpy(),
			"y_true": labels.cpu().numpy(),
			"val_loss": val_loss}

	def validation_epoch_end(self, outputs):
		# tn, fp, fn, tp
		# Evaluate metrics
		y_pred, y_true = [], []
		for batch in outputs:
			y_pred = np.concatenate((y_pred, batch["y_pred"]))
			y_true = np.concatenate((y_true, batch["y_true"]))

		auc_score = roc_auc_score(y_true.astype(int), y_pred)

		# Get the label here
		y_pred = (y_pred > self.threshold).astype(int)

		accuracy = accuracy_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred)
		recall = recall_score(y_true, y_pred)
		precision = precision_score(y_true, y_pred)

		tensorboard_logs = {
			"auc_score": auc_score,
			"accuracy_score": accuracy,
			"f1_score": f1,
			"recall_score": recall,
			"precision_score": precision
		}

		# self.logger.experiment.add_scalars("metrics", tensorboard_logs, self.current_epoch)
		# Add to one_cycle plot
		if (self.current_epoch + 1) % self.epoch_per_cycle == 0 and self.logger is not None:
			self.logger.experiment.add_scalars("one_cycle", tensorboard_logs, self.current_cycle)

		# Get the total loss of the validation
		total_loss = torch.stack([batch["val_loss"] for batch in outputs]).mean()
		self.val_log.append(total_loss.item())

		logs = {"val_loss": total_loss, "log": tensorboard_logs, "f1_score": f1}
		return logs

	def train_dataloader(self):
		return self.trainloader

	def val_dataloader(self):
		return self.valloader

	def get_channel(self):
		images, labels = next(iter(self.trainloader))
		_, channels, _, _ = images[0]
		return channels

	def get_max_epochs(self):
		return self.num_cycle * self.epoch_per_cycle