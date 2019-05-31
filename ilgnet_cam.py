# -*- coding: utf-8 -*-
"""ILGNet_CAM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TbfPphP6fgZDSKv7uVi8j1aeySnQ5u8g
"""

import os
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Colab Notebooks/ILGNet CAM')

from ILGNet_Model import *
import torch

import torch.nn as nn

from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2, torch

classes = {1: 'aesthetic', 0: 'not aesthetic'}

net =ILGNet_modified(num_classes=2,transform_input=False,init_weights=True)
chkpt = torch.load('ILGNet_AVA2_Model_Epoch_94_TrainAcc_91.373439_ValAcc_87.92314.pth',map_location='cpu')
net.load_state_dict(chkpt['state_dict'])

features_blobs0 = []
def hook_feature0(module, input, output):
    features_blobs0.append(output.data.cpu().numpy())

# net._modules.get(final_conv).register_forward_hook(hook_feature)
net.Maxpool1_1.register_forward_hook(hook_feature0)
# net.Maxpool2.register_forward_hook(hoo_feature)
# net.Inception_Global.register_forward_hook(hook_feature)k

features_blobs1 = []
def hook_feature1(module, input, output):
    features_blobs1.append(output.data.cpu().numpy())

# net._modules.get(final_conv).register_forward_hook(hook_feature)
# net.Maxpool1_1.register_forward_hook(hook_feature)
net.Maxpool2.register_forward_hook(hook_feature1)

features_blobs2 = []
def hook_feature2(module, input, output):
    features_blobs2.append(output.data.cpu().numpy())
    

# net._modules.get(final_conv).register_forward_hook(hook_feature)
# net.Maxpool1_1.register_forward_hook(hook_feature)
# net.Maxpool2.register_forward_hook(hook_feature)
net.Inception_Global.register_forward_hook(hook_feature2)
# features_blobs = np.concatenate((features_blobs0,features_blobs1,features_blobs2), axis = 0)

features_blobs = []
def hook_feature3(module, input, output):
    features_blobs.append(np.concatenate((features_blobs0[0],features_blobs1[0],features_blobs2[0]), axis = 1))

# net._modules.get(final_conv).register_forward_hook(hook_feature)
# net.Maxpool1_1.register_forward_hook(hook_feature)
# net.Maxpool2.register_forward_hook(hook_feature)
net.avgpool.register_forward_hook(hook_feature3)
# features_blobs = np.concatenate((f

# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
#     for idx in class_idx:
    print("feature_conv",feature_conv)
    print("weight_softmax",weight_softmax[class_idx])
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam(net, features_blobs, img_pil, classes, root_img):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
#     print(weight_softmax)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    h_x = F.softmax(logit).data.squeeze()
#     print(h_x)
    probs, idx = h_x.sort(0, True)
#     print(probs)
#     if(probs[0].item()>probs[1].item()):
#         class_pred = 0
#     else:
#         class_pred = 1 
#     # output: the prediction
#     for i in range(0, 2):
#         line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
#         print(line)
#     print(weight_softmax)
    CAMs = returnCAM(features_blobs[0], weight_softmax, idx[0].item())

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(root_img)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('cam4.jpg', result)

from PIL import Image
root = 'sample2.jpeg'
img = Image.open(root)
get_cam(net, features_blobs, img, classes, root)