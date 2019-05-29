#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:35:02 2019

@author: debopriyo
"""

from __future__ import print_function
import sys
import torch
from PIL import Image
import matplotlib.pyplot as pyplot
import numpy as np
import torchvision as tv
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable as V
from torch.autograd import Function as F
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from Datasets import ImageNet, AVA_Aesthetics_Ranking_Dataset
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
num_classes = 2
iterations = 100
lr = 0.0001
BatchSize = 200
#%%
path = "/home/debo/dataset/AVA/"
transform = transforms.Compose([transforms.ToTensor()])
                                
trainset = AVA_Aesthetics_Ranking_Dataset(root_dir=path,sample_type='train',transform=transform)
#trainset = datasets.CIFAR10(root='./CIFAR10',train=True,transform=transform,download=True)
trainset_len = trainset.__len__()
print('Trainset Length :', trainset_len)
trainsetloader = torch.utils.data.DataLoader(trainset,batch_size=BatchSize,shuffle=True,num_workers=8)
#valset = datasets.CIFAR10(root='./CIFAR10',train=False,transform=transform,download=True)
valset = AVA_Aesthetics_Ranking_Dataset(root_dir=path,sample_type='val',transform=transform)
valset_len = valset.__len__()
print('Valset Length :', valset_len)
valsetloader = torch.utils.data.DataLoader(valset,batch_size=BatchSize,shuffle=False,num_workers=8)
#%%
net = ILGNet(num_classes=1000,transform_input=False,init_weights=True)
chkpt = torch.load('/home/debo/ng_sural/data/ILGNet/models/ILGNet_ImageNet_Model_Epoch_2_TrainLoss_61.896224_ValAcc_54.302.pth',map_location='cpu')
net.load_state_dict(chkpt['state_dict'])
#%%
#modify the last fc_out layer
net.fc_out = nn.Linear(1024,num_classes)
#%%
for p in net.parameters():
    p.requires_grad_(False)

for p in net.Inception_Global.parameters():
    p.requires_grad_(True)
for p in net.Inception_Local2.parameters():
    p.requires_grad_(True)
for p in net.Conv_Global.parameters():
    p.requires_grad_(True)
#
#
net.fc_out.weight.requires_grad = True
net.fc_out.bias.requires_grad = True

net.fc_global1.weight.requires_grad = True
net.fc_global1.bias.requires_grad = True

net.fc_global2.weight.requires_grad = True
net.fc_global2.bias.requires_grad = True

net.fc_l1.weight.requires_grad = True
net.fc_l1.bias.requires_grad = True

net.fc_l2.weight.requires_grad = True
net.fc_l2.bias.requires_grad = True


#net = net.double()
#
#%%
#use_gpu = torch.cuda.is_available()
use_gpu = False
if use_gpu :
    print('GPU is available...')
    sys.stdout.flush()
    net = net.cuda()
else:
    print('GPU not available...')
    sys.stdout.flush()
    net = net.cpu()
#%%
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)
##optimizer = optim.SGD(net.parameters(), lr = lr, momentum=0.9)
criterion = nn.NLLLoss()
with open('/home/debo/ng_sural/data/ILGNet/models/ILGNet_AVA2_Model_Training_Validation.csv','w') as f:
    f.write('Epoch,LearningRate,AvgTrainLoss,AvgTrainAcc,AvgValAcc\n')
##%%
start = time.time()
valAcc = []
trainLoss = []
for epoch in range(iterations):
    
    epochStart = time.time()
    tr_runningLoss = 0.0
    val_runningLoss = 0.0
    tr_running_correct = 0
    print("Epoch:",epoch+1,"started...")
    sys.stdout.flush()
    for i, data in enumerate(trainsetloader, 0):
        net.train(True)
        inputs, labels = data

        inputs = inputs.float()

        #wrap them in variable
        if use_gpu:
          inputs, labels = V(inputs.cuda()),  V(labels.cuda())
        else:
          inputs,labels = V(inputs), V(labels)

        optimizer.zero_grad() # zeroes the gradient buffers of all parameters

        outputs = net(inputs) # forward

        loss = criterion(F.log_softmax(outputs, dim=1), labels) # calculate loss

        loss.backward() # backpropagate the loss

        optimizer.step()

        tr_runningLoss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        labels = labels.cpu()
        
        tr_running_correct += ((predicted==labels).sum()).item()
        print("Epoch :{:.0f} ; Batch : {:.0f} ; Training Loss : {:.6f} ; Training Accuracy : {:.6f}".format(epoch+1, i+1, tr_runningLoss/(i+1),(tr_running_correct/((i+1)*BatchSize))*100))
        sys.stdout.flush()
    avgTrainLoss = tr_runningLoss/(trainset_len/BatchSize)
    avgTrainAcc = (tr_running_correct/(trainset.__len__()))*100
    print("Epoch :{:.0f} ; Average Train Accuracy : {:.6f}".format(epoch+1,avgTrainAcc))
    sys.stdout.flush()
    trainLoss.append(avgTrainLoss)
#    if (epoch+1) % 5 == 0:
#        lr = lr * 0.1
#        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)


    print('Starting Validation...')
    sys.stdout.flush()
    with torch.no_grad():
        net.train(False) # For validation
        val_running_correct = 0
       
        for i, data in enumerate(valsetloader, 0):
            inputs, labels = data
#            labels = labels.to(device="cpu", dtype=torch.int64)
            
            inputs = inputs.float()
#            hsv_inputs = hsv_inputs.float()
            # Wrap them in Vriable
            if use_gpu:
                inputs = V(inputs.cuda())
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu()
            else:
                inputs = V(inputs)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
            val_running_correct += ((predicted == labels).sum()).item()
            
#            val_runningLoss += loss.item()
        avgValAcc = (val_running_correct/(valset.__len__()))*100
        
    valAcc.append(avgValAcc)
    print('Val Accuracy :',avgValAcc)
    epochEnd = time.time()-epochStart
    print('Iteration: {:.0f}/{:.0f} ; Training Loss: {:.6f} ; Training Accuracy : {:.6f} ;Validation Accuracy: {:.6f} ; Time Consumed: {:.0f}m {:.0f}s'.
          format(epoch + 1, iterations, avgTrainLoss, avgTrainAcc, avgValAcc, epochEnd//60, epochEnd%60))
    
    sys.stdout.flush()
    model_path = "/home/debo/ng_sural/data/ILGNet/models/ILGNet_AVA2_Model_Epoch_"+\
                    str(epoch+1)+"_TrainAcc_"+str(round(avgTrainAcc,6))+"_ValAcc_"+\
                    str(round(avgValAcc,6))+".pth"
    with open('/home/debo/ng_sural/data/ILGNet/models/ILGNet_AVA2_Model_Training_Validation.csv','a') as f:
        f.write(str(epoch+1)+','+str(lr)+','+str(avgTrainLoss)+','+str(avgTrainAcc)+','+str(avgValAcc)+'\n')
    torch.save({
            'epoch': epoch+1,
            'state_dict': net.state_dict()
            }, model_path)
    
end = time.time() - start
print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))
sys.stdout.flush()       
