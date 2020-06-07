import torch
import  torch.nn as nn
import torch.optim as optim
import math
from models.dualattention import DualAttentionModel
from trainer.trainer import TrainerSkeleton

torch.backends.cudnn.benchmark = True

class DualAttentionTrainer(TrainerSkeleton):
		def __init__(self, trainloader, valloader,
								valset,
								netname,
								cool_down = 0,
								num_classes = 2,
								base_lr = 1e-3,
								max_lr = 1e-2,
								num_cycle = 1,
								epoch_per_cycle = 4,
								running_scheduler = True
								):
				super(DualAttentionTrainer, self).__init__(
										trainloader = trainloader,
										valloader = valloader,
										valset = valset,
										netname = netname)

				# Load pretrained model
				in_channel = 512
				self.model = DualAttentionModel(in_channel = in_channel, multi_grid = True, multi_dilation = [4, 8, 16])

				# From fc to correct outputs
				self.num_classes = num_classes
				self.dropout = nn.Dropout(0.2)
				self.outputs = nn.Linear(in_channel, num_classes -1)
				self.init_weight()

				# Hyperparameters
				self.running_scheduler = True
				self.base_lr = base_lr
				self.max_lr = max_lr
				self.num_cycle = num_cycle
				self.epoch_per_cycle = epoch_per_cycle
				self.current_cycle = 0
				self.cool_down = cool_down

		def init_weight(self):
				n = 1000 * self.num_classes
				nn.init.normal_(self.outputs.weight, 0, 1 / math.sqrt(n))

		def forward(self, inputs):
			logits = self.model(inputs)
			logits = self.dropout(logits)
			outputs = self.outputs(logits)
			return outputs

		def configure_optimizers(self):
			print(">> Reset the optimizer")
			self.optimizer = optim.SGD(self.model.parameters(), lr = self.base_lr, weight_decay = 1e-5, momentum = 0.9)

			self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
													max_lr = self.max_lr,
													epochs = self.epoch_per_cycle,
													steps_per_epoch = len(self.trainloader),
													div_factor = 25,
													final_div_factor = 1e4
													)

			return self.optimizer

		def on_epoch_end(self):
			if self.logger:
				avg_train = sum(self.training_log) / len(self.trainloader)
				# avg_val = sum(self.val_log) / self.epoch_per_cycle

				self.logger.experiment.add_scalar("one_cycle/training_loss", avg_train, self.current_cycle)
				# self.logger.experiment.add_scalar("one_cycle/val_loss", avg_val, self.current_cycle)
				self.training_log = []

			if (self.current_epoch + 1) % self.epoch_per_cycle == 0:
					# This is only for 1 cycle
					self.current_cycle = (self.current_epoch + 1) // self.epoch_per_cycle
					self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
															max_lr = self.max_lr,
															epochs = self.epoch_per_cycle,
															steps_per_epoch = len(self.trainloader))