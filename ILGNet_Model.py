
from __future__ import print_function
import sys
import torch
#from PIL import Image
#import matplotlib.pyplot as pyplot
import numpy as np
import torchvision as tv
#from torchvision import datasets
#import torchvision.transforms as transforms
#from torch.autograd import Variable as V
#from torch.autograd import Function as F
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#import time
#from Datasets import ImageNet, AVA_Aesthetics_Ranking_Dataset
#%%
class ILGNet(nn.Module):
    def __init__(self, num_classes=None, transform_input=None, init_weights=True):
        super(ILGNet, self).__init__()
        self.transform_input = transform_input
        self.Pretreatment = nn.Sequential(
                                BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                                nn.BatchNorm2d(64, eps=0.001),
                                BasicConv2d(64, 64, kernel_size=1, padding=0),
                                BasicConv2d(64, 192, kernel_size=3, padding=1),
                                nn.BatchNorm2d(192, eps=0.001))
    
        self.Maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        
        self.Inception_Local1 = Inception(192, 64, 96, 128, 16, 32, 32)
        
        self.Inception_Local2 = Inception(256, 128, 128, 192, 32, 96, 64)
        
        self.Maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.Inception_Global = Inception(480, 192, 96, 208, 16, 48, 64)
    
        self.avgPool = nn.AvgPool2d(5,stride=3,padding=0)
        
        self.Conv_Global = BasicConv2d(512, 512, kernel_size=1, stride=1, padding=1)
        
        self.fc_l1 = nn.Linear(256*28*28, 256)
        self.dropout_local1 = nn.Dropout(0.5)
        self.fc_l2 = nn.Linear(480*28*28, 256)
        self.dropout_local2 = nn.Dropout(0.5)
        
        self.fc_global1 = nn.Linear(512*6*6, 512)
        self.dropout_global1 = nn.Dropout(0.5)
        self.fc_global2 = nn.Linear(512,512)
        self.dropout_global2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-1, 1, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # N x 3 x 224 x 224
        x = self.Pretreatment(x)
#        print(x.size())
        # N x 192 x 56 x 56
        x = self.Maxpool1(x)
#        print(x.size())
        # N x 192 x 28 x 28
        x1 = self.Inception_Local1(x)
#        print(x1.size())
        # N x 256 x 28 x 28
        x2 = self.Inception_Local2(x1)
#        print(x2.size())
        # N x 480 x 28 x 28
        x3 = self.Maxpool2(x2)
#        print(x3.size())
        # N x 480 x 14 x 14
        x3 = self.Inception_Global(x3)
#        print(x3.size())
        # N x 512 x 14 x 14
        x3 = self.avgPool(x3)
        x3 = self.Conv_Global(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc_global1(x3)
        x3 = self.dropout_global1(x3)
        x3 = self.fc_global2(x3)
        x3 = self.dropout_global2(x3)
        
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc_l2(x2)
        x2 = self.dropout_local2(x2)
        
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc_l1(x1)
        x1 = self.dropout_local1(x1)
        
        out = torch.cat([x1,x2,x3],1)
        
        out = self.fc_out(out)
#        out = F.softmax(out, dim=1)
        
        return out
    
class ILGNet_modified(nn.Module):
    def __init__(self, num_classes=None, transform_input=None, init_weights=True):
        super(ILGNet_modified, self).__init__()
        self.transform_input = transform_input
        self.Pretreatment = nn.Sequential(
                                BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                                nn.BatchNorm2d(64, eps=0.001),
                                BasicConv2d(64, 64, kernel_size=1, padding=0),
                                BasicConv2d(64, 192, kernel_size=3, padding=1),
                                nn.BatchNorm2d(192, eps=0.001))
    
        self.Maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        
        self.Inception_Local1 = Inception(192, 64, 96, 128, 16, 32, 32)
        
        self.Maxpool1_1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        
        self.Inception_Local2 = Inception(256, 128, 128, 192, 32, 96, 64)
        
        self.Maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.Inception_Global = Inception(480, 192, 96, 208, 16, 48, 64)
    
#        self.avgPool = nn.AvgPool2d(5,stride=3,padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
#        self.Conv_Global = BasicConv2d(512, 512, kernel_size=1, stride=1, padding=1)
        
#        self.fc_l1 = nn.Linear(256*28*28, 256)
#        self.dropout_local1 = nn.Dropout(0.5)
#        self.fc_l2 = nn.Linear(480*28*28, 256)
#        self.dropout_local2 = nn.Dropout(0.5)
        
#        self.fc_global1 = nn.Linear(512*6*6, 512)
#        self.dropout_global1 = nn.Dropout(0.5)
#        self.fc_global2 = nn.Linear(512,512)
#        self.dropout_global2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(1248, num_classes)
        if init_weights:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-1, 1, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # N x 3 x 224 x 224
        x = self.Pretreatment(x)
#        print(x.size())
        # N x 192 x 56 x 56
        x = self.Maxpool1(x)
#        print(x.size())
        # N x 192 x 28 x 28
        x1 = self.Inception_Local1(x)
#        print(x1.size())
        # N x 256 x 28 x 28
        x1_max = self.Maxpool1_1(x1)
        # N x 256 x 14 x 14
        x2 = self.Inception_Local2(x1)
#        print(x2.size())
        # N x 480 x 28 x 28
        x2_max = self.Maxpool2(x2)
#        print(x3.size())
        # N x 480 x 14 x 14
        x3 = self.Inception_Global(x2_max)
#        print(x3.size())
        # N x 512 x 14 x 14

        
        out = torch.cat([x1_max,x2_max,x3],1)
#        print(out.size())
        out = self.avgpool(out)
        # N x 1248 x 1 x 1
        out = out.view(out.size(0), -1)
        # N x 1024
        out = self.dropout(out)
        out = self.fc_out(out)

        
        return out

class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
#        print('Branch1 :',branch1.size())
        branch2 = self.branch2(x)
#        print('Branch2 :',branch1.size())
        branch3 = self.branch3(x)
#        print('Branch3 :',branch1.size())
        branch4 = self.branch4(x)
#        print('Branch4 :',branch1.size())

        outputs = [branch1, branch2, branch3, branch4]
#        print(outputs.size())
        x = torch.cat(outputs, 1)
#        print(x.size())
        return x
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
#%%
