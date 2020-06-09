'''
  SKNet idea is from: https://arxiv.org/abs/1903.06586
  We implemented this code from our understanding from the paper
    In addition, we also refer to ResNeXt from PyTorch to implement
    the dilation and group channels operation.
'''


from typing import Match, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SKAttention(nn.Module):
	def __init__(self, kernel_size,
				in_planes,
				r,
				L = 32,
				is_debug = False
				):

		super().__init__()
		self.is_debug = is_debug
		self.d = max(in_planes // r, L)
		self.z = nn.Sequential(
			nn.Linear(in_planes, self.d),
			# ! Instance Norm
			nn.BatchNorm1d(self.d),
			nn.ReLU()
		)
		self.projection = nn.ModuleList()
		for i in range(kernel_size):
			self.projection.add_module("projection_{0}x{0}".format(i), nn.Linear(self.d, in_planes))

	def forward(self, inputs):
		batch_size, channels, _, _ = inputs.shape
		flatten = torch.mean(inputs.view(batch_size, channels, -1), dim = 2)
		features = self.z(flatten)

		# outputs = torch.Tensor()
		outputs = []
		for layer in self.projection:
			outputs.append(layer(features))
			# outputs = torch.stack((outputs, layer(features)), dim = 0)

		outputs = torch.stack(outputs, dim = 0)
		outputs = F.softmax(outputs, dim = 0)

		return outputs

class SKConvolution(nn.Module):
	def __init__(self,
							in_planes,
							kernel_size,
							stride,
							cardinality,
							r,
							is_debug = False
							):
		super().__init__()
		self.is_debug = is_debug
		self.kernel_size = kernel_size

		self.conv_blocks = nn.ModuleList()
		for kernel in kernel_size:
			self.conv_blocks.add_module("layer_{0}x{0}".format(kernel),
				nn.Sequential(
					nn.Conv2d(in_planes, in_planes, kernel_size = kernel,
										padding = (kernel - 1) // 2, stride = stride, groups = cardinality, bias = False),
					nn.BatchNorm2d(in_planes),
					nn.ReLU()
				)
			)
			# self.conv_blocks.append(
			# 	nn.Sequential(
			# 	nn.Conv2d(in_planes, in_planes, kernel_size = kernel,
			# 						padding = (kernel - 1) // 2, stride = stride, groups = cardinality, bias = False),
			# 	nn.BatchNorm2d(in_planes),
			# 	nn.ReLU()
			# 	)
			# )

		self.attention_block = SKAttention(kernel_size = len(kernel_size), in_planes = in_planes, r = r)

	def forward(self, inputs):
		features = []
		for layer in self.conv_blocks:
			features.append(layer(inputs))

		features = torch.stack(features)
		fused_features = torch.sum(features, dim = 0)

		# Stack: (kernel_size, batch, channel)
		attention = self.attention_block(fused_features)

		outputs = attention[:, :, :, None, None] * features
		outputs = torch.sum(outputs, dim = 0)


		# if self.is_debug:
		# 	print(">> Check if fused_features = sum(features) along the first dimension")
		# 	channel1 = features[0].unsqueeze(dim = 0)
		# 	channel2 = features[1].unsqueeze(dim = 0)
		# 	temp = fused_features.unsqueeze(dim = 0)
		# 	print(torch.all(torch.eq(channel2 + channel1, temp)))

		# 	print("\n>> Check if outputs is the element-wise multiplication between channels and image")
		# 	temp = torch.ones(attention.shape) * 2
		# 	out_temp = temp[:, :, :, None, None] * features
		# 	print((out_temp[0, 0, 0] == features[0, 0, 0] * 2).all())

		return outputs

class SKBlock(nn.Module):
	def __init__(self, in_planes, bottleneck_width, cardinality, stride, expansion, r, is_debug = False):
		super().__init__()
		inner_width = cardinality * bottleneck_width
		self.cardinality = cardinality
		self.expansion = expansion

		self.basic_block = nn.Sequential(
			nn.Conv2d(in_planes, inner_width, kernel_size = 1, bias = False),
			nn.BatchNorm2d(inner_width),
			nn.ReLU(),
			SKConvolution(inner_width, [3, 5], stride, cardinality, r, is_debug = is_debug),
			nn.BatchNorm2d(inner_width),
			nn.ReLU(),
			nn.Conv2d(inner_width, inner_width * expansion, kernel_size = 1, bias = False),
			nn.BatchNorm2d(inner_width * expansion)
		)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != inner_width * self.expansion:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, inner_width * self.expansion, kernel_size = 1, stride = stride, bias = False)
			)

		self.batch_norm = nn.BatchNorm2d(inner_width * self.expansion)

	def forward(self, inputs):
		# inputs = torch.cat((inputs, inputs), dim = 0)
		features = self.basic_block(inputs)
		outputs = features + self.shortcut(inputs)
		outputs = self.batch_norm(outputs)
		return outputs

class SKNet(nn.Module):
	def __init__(self,
							layers,
							cardinality,
							bottleneck_width,
							r,
							expansion = 2,
							num_classes = 2,
							complete_model = True
							):

	 super(SKNet, self).__init__()

	 # Define necessary elements for Net
	 self.in_planes = 64
	 self.cardinality = cardinality
	 self.bottleneck_width = bottleneck_width
	 self.r = r
	 self.expansion = expansion

	 # Stem head
	 self.stem_head = nn.Sequential(
		 nn.Conv2d(in_channels = 3, out_channels = self.in_planes, kernel_size = 7, stride = 2, padding = 2),
		 nn.BatchNorm2d(self.in_planes),
		 nn.ReLU(),
		 nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
	 )

	 self.layer_1 = self._make_layers(layers[0], 1)
	 self.layer_2 = self._make_layers(layers[1], 2)
	 self.layer_3 = self._make_layers(layers[2], 2)
	 self.layer_4 = self._make_layers(layers[3], 2)

	 '''
	 	TODO: Consider to add dropout
	 '''
	 self.fc = nn.Linear(self.in_planes, num_classes - 1)

	 self.init_weight()

	def init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2.0 / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)

	def _make_layers(self, blocks, stride):
		strides = [stride] + [1] * (blocks - 1)
		layers = []
		for stride in strides:
			layers.append(
				SKBlock(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion, self.r)
			)
			self.in_planes =  self.expansion * self.bottleneck_width * self.cardinality

		self.bottleneck_width *= 2
		return nn.Sequential(*layers)

	def forward(self, inputs):
		features = self.stem_head(inputs)
		features = self.layer_1(features)
		features = self.layer_2(features)
		features = self.layer_3(features)
		features = self.layer_4(features)

		# Flatten the features to fully connected layer
		outputs = nn.AvgPool2d(7)(features)
		outputs = outputs.view(inputs.shape[0], -1)
		outputs = self.fc(outputs)
		return outputs


def sknet18_32x4d(num_classes):
	'''
		TODO: add different kernel size
	'''
	return SKNet(layers = [2, 2, 2, 2], cardinality = 32, bottleneck_width = 4, r = 16, num_classes = num_classes)