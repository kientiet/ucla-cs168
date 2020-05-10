'''
    This implementation was based on https://github.com/wielandbrendel/bag-of-local-features-models/
    with slight variation
'''
import torch
import torch.nn as nn
import torchvision
import yaml
import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False) # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:,:,:-diff,:-diff]
        
        out += residual
        out = self.relu(out)

        return out

class BagNetArchitecture(nn.Module):
    def __init__(self, 
                in_channel, 
                net_type = "bagnet-9", 
                avg_pool = True,
                BottleNeck = Bottleneck,
                num_classes = 2):

        super(BagNetArchitecture, self).__init__()
        self.in_channel = in_channel
        self.inplanes = 64
        self.type = net_type

        self.stem_heads = nn.Sequential(
            nn.Conv2d(in_channel, self.inplanes, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size = 3, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(self.inplanes, momentum = 0.001),
            nn.ReLU()
        )

        layers, strides, kernel3 = self.load_yaml(net_type)
        self.layer1 = self._make_layer(BottleNeck, 64, layers[0], stride = strides[0], kernel3 = kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(BottleNeck, 128, layers[1], stride = strides[1], kernel3 = kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(BottleNeck, 256, layers[2], stride = strides[2], kernel3 = kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(BottleNeck, 512, layers[3], stride = strides[3], kernel3 = kernel3[3], prefix='layer4')        

        self.avgpool = nn.AvgPool2d(1, stride = 1)
        self.fc = nn.Linear(512 * BottleNeck.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = BottleNeck

        self.init_weight()

    
    def load_yaml(self, net_type):
        with open("models/bagnet.yml") as yaml_file:
            architecture = yaml.load(yaml_file, Loader = yaml.FullLoader)[net_type]
            layers = architecture["layers"]
            strides = architecture["strides"]
            kernel3 = architecture["kernels"]
        
        return layers, strides, kernel3


    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride = 1, kernel3 = 0, prefix = ""):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size = kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size = kernel))
        
        return nn.Sequential(*layers)
    
    def forward(self, inputs):
        inputs = self.stem_heads(inputs)
        inputs = self.layer1(inputs)
        inputs = self.layer2(inputs)
        inputs = self.layer3(inputs)
        inputs = self.layer4(inputs)
        
        if self.avg_pool:
            inputs = nn.AvgPool2d(inputs.size()[2], stride=1)(inputs)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = self.fc(inputs)
        else:
            inputs = inputs.permute(0,2,3,1)
            outputs = self.fc(inputs)
        
        return outputs