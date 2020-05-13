import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# from models.bagnetorg import bagnet33
class DualAttentionModel(nn.Module):
    def __init__(self, in_channel, num_classes = 2, model = "resnet18"):
        super().__init__()
        if model == "resnet18":
            self.model = torchvision.models.resnet18(pretrained = True)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        else:
            '''
                TODO: Check if the architecture is cool
            '''
            self.model = bagnet33(pretrained = True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])

        '''
            TODO: check this
        '''
        out_channel = in_channel
        self.dahead = DualAttentionHead(in_channel, out_channel = out_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    '''
        TODO: Initialize the weights
    '''

    def forward(self, inputs):
        '''
            TODO: 
            1/ what is difference between nn.AvgPool and nn.AdaptiveAvgPool
            2/ using transformer for adaptive pooling?

        '''
        features = self.model(inputs)
        attention = self.dahead(features)
        attention_flatten = self.avg_pool(attention)
        outputs = attention_flatten.view(inputs.shape[0], -1)
        return outputs


class DualAttentionHead(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        '''
            TODO: check different type of inter_channel here
        '''
        self.inter_channel = in_channel // 4

        self.stem_head = nn.Sequential(
            nn.Conv2d(in_channel, self.inter_channel, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )

        self.position_attention = PointAttentionBlock(self.inter_channel)
        self.position_fused = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size = 1)
        )

        self.channel_attention = ChannelAttentionBlock(self.inter_channel)
        self.channel_fused = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size = 1)
        )

        self.outputs = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(self.inter_channel, out_channel, kernel_size = 1)
        )
    
    def forward(self, inputs):
        inputs_fused = self.stem_head(inputs)

        position = self.position_attention(inputs_fused)
        position_fused = self.position_fused(position)

        channel = self.channel_attention(inputs_fused)
        channel_fused = self.channel_fused(channel)

        sum_fused = position_fused + channel_fused
        outputs = self.outputs(sum_fused)

        return outputs

class PointAttentionBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        '''
            TODO: Why divide 8
        '''
        self.project_query = nn.Conv2d(in_channel, in_channel // 8, kernel_size = 1)
        self.project_key = nn.Conv2d(in_channel, in_channel // 8, kernel_size = 1)
        self.project_value = nn.Conv2d(in_channel, in_channel, kernel_size = 1)

        self.softmax = nn.Softmax(dim = -1)
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def forward(self, inputs):
        batch_size, channels, height, width = inputs.shape

        # Project the inputs
        query = self.project_query(inputs)
        key = self.project_key(inputs)
        value = self.project_value(inputs)

        # Calculate the attention
        query = query.view(batch_size, -1, height * width)
        query = query.permute(0, 2, 1)
        key = key.view(batch_size, -1, height * width)
        attention_weight = torch.bmm(query, key)        
        attention_weight = self.softmax(attention_weight)

        # Get soft attention weight
        value = value.view(batch_size, -1, height * width)
        attention = torch.bmm(value, attention_weight.permute(0, 2, 1))
        outputs = self.alpha * attention.view(inputs.shape) + inputs

        return outputs

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.softmax = nn.Softmax(dim = -1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.shape
        
        # Get the proper form
        query = inputs.view(batch_size, channels, -1)
        key = inputs.view(batch_size, channels, -1).permute(0, 2, 1)
        value = inputs.view(batch_size, channels, -1)

        # Calculate attention
        attention_weight = torch.bmm(query, key)
        '''
            TODO: check this
        '''
        attention_weight = torch.max(attention_weight, -1, keepdim = True)[0].expand_as(attention_weight) - attention_weight 
        attention_weight = self.softmax(attention_weight)

        # Get soft attention
        value = value.view(batch_size, channels, -1)
        outputs = torch.bmm(attention_weight, value)
        outputs = self.gamma * outputs.view(inputs.shape) + inputs

        return outputs