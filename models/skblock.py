import torch
import torch.nn as nn
import torch.nn.functional as F

class SKBlock(nn.Module):
	def __init__(self, in_channel):
		super().__init__()
		self.stem_head = nn.Conv2d(in_channel, in_channel, kernel_size = 1, bias = False)
		'''
			TODO: Adding stride?
		'''
		self.sk_conv = SKConvolution(in_channel, kernel_size = [3, 5])
	
	def forward(self, inputs):
		stem = self.stem_head(inputs)
		outputs = self.sk_conv(stem)
		return outputs

class SKConvolution(nn.Module):
	def __init__(self, in_channel, kernel_size):
		super().__init__()
		self.kernel_size = kernel_size
		
		out_channel = in_channel
		self.conv_blocks = []
		'''
			TODO: Adding the group convolution
		'''
		for kernel in kernel_size:
			self.conv_blocks.append(
				nn.Sequential(
					nn.Conv2d(in_channel, out_channel, kernel_size = kernel, padding = (kernel - 1) // 2, bias = False),
					nn.BatchNorm2d(out_channel),
					nn.ReLU()
				)
			)
	
	def forward(self, inputs):
		features = []
		for layer in self.conv_blocks:
			features.append(layer(inputs))

		features = torch.stack(features)
		features = torch.sum(features, axis = 0)
		print(features.shape)

class SKAttention(nn.Module):
	def __init__(self, kernel_size, num_channel, groups, L = 32):
		super().__init__()
		self.d = max(num_channel // groups, L)
		self.z = nn.Sequential(
			nn.Linear(num_channel, self.d),
			nn.BatchNorm2d(self.d),
			nn.ReLU()
		)
		self.projection = []
		for _ in range(kernel_size):
			self.projection.append(nn.Linear(num_channel, self.d))

	def forward(self, inputs):
		batch_size, channels, _, _ = inputs.shape
		flatten = torch.mean(inputs.view(batch_size, channels, -1), dim = 2)
		features = self.z(flatten)

		outputs = []
		for layer in self.projection:
			outputs.append(layer(features))
		
		outputs = torch.stack(outputs, dim = 0)
		outputs = F.softmax(outputs, dim = 0)
		print(outputs.shape)

class Transformer(nn.Module):
	def __init__(self, in_channel):
		super().__init__()
		out_channel = in_channel
		self.project_query = nn.Conv2d(in_channel = in_channel, out_channel = in_channel, kernel_size = )

inputs = torch.rand(10, 20, 224, 224)
net = SKBlock(20)
net(inputs)