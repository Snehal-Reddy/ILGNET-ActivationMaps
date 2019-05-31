from __future__ import print_function
import sys
import torch
#from PIL import Image
import matplotlib.pyplot as pyplot
import numpy as np
#import torchvision as tv
#from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable as V
from torch.autograd import Function as F
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from Datasets import AVA_Aesthetics_Ranking_Dataset
from ILGNet_Model import ILGNet,ILGNet_modified
#%%
num_classes = 2
iterations = 100
lr = 0.0001
BatchSize = 200
IMG_SIZE=224
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
net = ILGNet(num_classes=num_classes,transform_input=False,init_weights=False)
net_modified =ILGNet_modified(num_classes=num_classes,transform_input=False,init_weights=True)
chkpt = torch.load('/home/debo/ng_sural/data/ILGNet/models/ILGNet_AVA2_Model_Epoch_2_TrainAcc_85.596305_ValAcc_85.434196.pth',map_location='cpu')
net.load_state_dict(chkpt['state_dict'])
#layers=[x.data for x in net.parameters()]
net_modified.Pretreatment=net.Pretreatment
net_modified.Maxpool1=net.Maxpool1
net_modified.Inception_Local1=net.Inception_Local1
net_modified.Inception_Local2=net.Inception_Local2
net_modified.Maxpool2=net.Maxpool2
net_modified.Inception_Global=net.Inception_Global
#
for p in net_modified.parameters():
    p.requires_grad_(True)
#
#for p in net_modified.Pretreatment.parameters():
#    p.requires_grad_(False)
#for p in net_modified.Inception_Local1.parameters():
#    p.requires_grad_(False)

#for param in net_modified.parameters():
#    print(param)
#    break
use_gpu = torch.cuda.is_available()
#use_gpu = False
if use_gpu :
    print('GPU is available...')
    sys.stdout.flush()
    net_modified = net_modified.cuda()
else:
    print('GPU not available...')
    sys.stdout.flush()

#%%
optimizer = optim.Adam(net_modified.parameters(), lr = lr, weight_decay=0.0005)
#optimizer = optim.SGD(net_modified.parameters(), lr = lr, momentum=0.9, weight_decay=0.0005)

criterion = nn.NLLLoss()
with open('/home/debo/ng_sural/data/ILGNet_modified/models/ILGNet_AVA2_Model_Training_Validation.csv','w') as f:
    f.write('Epoch,LearningRate,AvgTrainLoss,AvgTrainAcc,AvgValAcc\n')
#%%
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
        net_modified.train(True)
        inputs, labels = data
#        labels = labels.to(device="cpu", dtype=torch.int64)
        inputs = inputs.float()
#        hsv_inputs = hsv_inputs.float()
        #wrap them in variable
        if use_gpu:
          inputs, labels = V(inputs.cuda()),  V(labels.cuda())
        else:
          inputs,labels = V(inputs), V(labels)

        optimizer.zero_grad() # zeroes the gradient buffers of all parameters

        outputs = net_modified(inputs) # forward

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
    if (epoch+1) % 5 == 0:
        lr = lr * 0.1
        optimizer = optim.Adam(net_modified.parameters(), lr = lr)


    print('Starting Validation...')
    sys.stdout.flush()
    with torch.no_grad():
        net_modified.train(False) # For validation
        val_running_correct = 0
       
        for i, data in enumerate(valsetloader, 0):
            inputs, labels = data
#            labels = labels.to(device="cpu", dtype=torch.int64)
            
            inputs = inputs.float()
#            hsv_inputs = hsv_inputs.float()
            # Wrap them in Vriable
            if use_gpu:
                inputs = V(inputs.cuda())
                outputs = net_modified(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu()
            else:
                inputs = V(inputs)
                outputs = net_modified(inputs)
                _, predicted = torch.max(outputs.data, 1)
            val_running_correct += ((predicted == labels).sum()).item()
            
            val_runningLoss += loss.item()
        avgValAcc = (val_running_correct/(valset.__len__()))*100
        
    valAcc.append(avgValAcc)
    print('Val Accuracy :',avgValAcc)
    epochEnd = time.time()-epochStart
    print('Iteration: {:.0f}/{:.0f} ; Training Loss: {:.6f} ; Training Accuracy : {:.6f} ;Validation Accuracy: {:.6f} ; Time Consumed: {:.0f}m {:.0f}s'.
          format(epoch + 1, iterations, avgTrainLoss, avgTrainAcc, avgValAcc, epochEnd//60, epochEnd%60))
    
    sys.stdout.flush()
    model_path = "/home/debo/ng_sural/data/ILGNet_modified/models/ILGNet_AVA2_Model_Epoch_"+\
                    str(epoch+1)+"_TrainAcc_"+str(round(avgTrainAcc,6))+"_ValAcc_"+\
                    str(round(avgValAcc,6))+".pth"
    with open('/home/debo/ng_sural/data/ILGNet_modified/models/ILGNet_AVA2_Model_Training_Validation.csv','a') as f:
        f.write(str(epoch+1)+','+str(lr)+','+str(avgTrainLoss)+','+str(avgTrainAcc)+','+str(avgValAcc)+'\n')
    torch.save({
            'state_dict': net_modified.state_dict()
            }, model_path)
    
end = time.time() - start
print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))
sys.stdout.flush()  