
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2
from random import sample
from scipy import stats
import torch.nn.functional as F
#%%
def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img
def resizeAndCrop(img, size=(256,256)):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # scaling and cropping
    if aspect > 1: # horizontal image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        extra_region_w = np.floor((new_w-sw)/2).astype(int)
        # scale and crop
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = scaled_img[0:new_h,extra_region_w:new_w-extra_region_w]
    elif aspect < 1: # vertical image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        extra_region_h = np.floor((new_h-sh)/2).astype(int)
        # scale and crop
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = scaled_img[extra_region_h:new_h-extra_region_h,0:new_w]

    else: # square image
        new_h, new_w = sh, sw
        
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
    scaled_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return scaled_img
#%%

def randomCropResize(img,old_size=256,new_size=224):
    remove_reg=int((old_size-new_size)/2)
    r = np.random.choice([1,2,3,4,5,6])
    if r==1: #top left
        return img[0:new_size,0:new_size]
    elif r==2: #top right
        return img[0:new_size,remove_reg*2:old_size]
    elif r==3: #bottom left
        return img[remove_reg*2:old_size,0:new_size]
    elif r==4: #bottom right
        return img[remove_reg*2:old_size,remove_reg*2:old_size]
    elif r==5: # center crop
        return img[remove_reg:new_size+remove_reg,remove_reg:new_size+remove_reg]
    else: # resize
        return cv2.resize(img,(new_size,new_size), interpolation = cv2.INTER_AREA)
    
#%%
def img_mean_stddev(dataset):
    sum_img = [0,0,0]
    n = len(dataset)
    h , w = dataset[0][0][:2]
    N = n * h * w
    for img,label in dataset:
        sum_img[0] += np.sum(img[:,:,0])
        sum_img[1] += np.sum(img[:,:,1])
        sum_img[2] += np.sum(img[:,:,2])
    mean_val = [np.round(sum_img[0]/(N),2), np.round(sum_img[1]/(N),2), np.round(sum_img[2]/(N),2)]
    print(mean_val)          
    x_m_sq_sum = [0,0,0]
    for img,label in dataset:
        x_m_sq_sum[0] += np.sum((img[:,:,0] - mean_val[0])**2)
        x_m_sq_sum[1] += np.sum((img[:,:,1] - mean_val[1])**2)
        x_m_sq_sum[2] += np.sum((img[:,:,2] - mean_val[2])**2)
    
    stddev_val = [np.round(np.sqrt(x_m_sq_sum[0]/(N)),2), 
                  np.round(np.sqrt(x_m_sq_sum[1]/(N)),2), 
                  np.round(np.sqrt(x_m_sq_sum[2]/(N)),2)]
    print(stddev_val)
    return mean_val, stddev_val
#%%
class ImageNet(Dataset):
    def __init__(self, root_dir, sample_type=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_files = []
        self.labels=[]
        self.train_img_mean = []
        self.train_img_std = []
        
        if sample_type == 'train':
            data = pd.read_csv(self.root_dir+"devkit/data/train_data.csv")
            self.img_files = [self.root_dir+x for x in list(data['img_path'])]
            self.labels = list(data['label'])
        elif sample_type == 'val':
            self.img_files = [self.root_dir+"img_val/"+x for x in os.listdir(self.root_dir+"img_val") if ".JPEG" in x]
            self.img_files.sort()
            self.labels = list(pd.read_csv(self.root_dir+"devkit/data/ILSVRC2012_validation_ground_truth.txt",sep=",",header=None, index_col=False,names=["label"])['label'])
#            print(self.img_files[0])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        img = cv2.imread(self.img_files[idx])
#        img = np.asarray(img)
        img = resizeAndCrop(img,size=(256,256))
        
        
#        img = cv2.imread(self.img_files[idx])
#        
        if self.transform:
            img = randomCropResize(img,old_size=256,new_size=224)#random crop or resize
            if np.random.rand()>0.5:#random horizontal flip
                img = cv2.flip(img,0)
#            img = img.astype('float32')
            img = img / 255.0
#            print(img.shape)
#            img = transforms.ToPILImage()(torch.tensor(img))
            img[:,:,2] = (img[:,:,2] - 0.485)/0.229  
            img[:,:,1] = (img[:,:,1] - 0.456)/0.224
            img[:,:,0] = (img[:,:,0] - 0.406)/0.225
            
            #img[:,:,2] = (img[:,:,2] - 0.485) 
            #img[:,:,1] = (img[:,:,1] - 0.456)
            #img[:,:,0] = (img[:,:,0] - 0.406)
#            
#            img[:,:,0] = (img[:,:,0]-img[:,:,0].min())/(img[:,:,0].max()-img[:,:,0].min())
#            img[:,:,1] = (img[:,:,1]-img[:,:,1].min())/(img[:,:,1].max()-img[:,:,1].min())
#            img[:,:,2] = (img[:,:,2]-img[:,:,2].min())/(img[:,:,2].max()-img[:,:,2].min()) 
            img = self.transform(img)
        label = self.labels[idx]-1
        
        return img,label

#%%
class AVA_Aesthetics_Ranking_Dataset(Dataset):
    def __init__(self, root_dir, sample_type=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sample_type=sample_type
        self.labels = []
        self.files = []
        if sample_type=='train':
#            data=pd.read_csv(self.root_dir+"AVA_dataset/AVA_mean_rating_samples_top_10pc_bottom_10pc_train.csv")
            data=pd.read_csv(self.root_dir+"AVA_dataset/ILGNet/AVA2/train_ilgnet.txt", sep=" ")

#            self.labels = list(data.mean_rating)
#            self.files = list(data.img_id)
            self.labels = list(data.label)
            self.files = list(data.img)

            
        elif sample_type=='val':
#            data=pd.read_csv(self.root_dir+"AVA_dataset/AVA_mean_rating_samples_top_10pc_bottom_10pc_test.csv")
            data=pd.read_csv(self.root_dir+"AVA_dataset/ILGNet/AVA2/val.txt", sep=" ")


#            self.labels = list(data.mean_rating)
#            self.files = list(data.img_id)
            self.labels = list(data.label)
            self.files = list(data.img)


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
#        try:
#        if self.sample_type=='train':
#            img = cv2.imread(os.path.join(self.root_dir,"top_10pc_bottom_10pc_rated_resized_images_224_224_padded_black/train/"+str(self.files[idx])+'.jpg'))
#        elif self.sample_type=='val':
#            img = cv2.imread(os.path.join(self.root_dir,"top_10pc_bottom_10pc_rated_resized_images_224_224_padded_black/test/"+str(self.files[idx])+'.jpg'))
        
#        if self.sample_type=='train':
#            img = cv2.imread(os.path.join(self.root_dir,"images/"+str(self.files[idx])+'.jpg'))
#        elif self.sample_type=='val':
#            img = cv2.imread(os.path.join(self.root_dir,"images/"+str(self.files[idx])+'.jpg'))
        if self.sample_type=='train':
            img = cv2.imread(os.path.join(self.root_dir,"images/"+str(self.files[idx])))
        elif self.sample_type=='val':
            img = cv2.imread(os.path.join(self.root_dir,"images/"+str(self.files[idx])))

        img = resizeAndCrop(img,size=(224,224))
#        img = np.asarray(img)
#        if self.sample_type=='train':
#            img = resizeAndCrop(img,size=(256,256))
#        else:
#            img = resizeAndCrop(img,size=(224,224))
        
    
        if self.transform:
#            if self.sample_type=='train':
#                img = randomCropResize(img,old_size=256,new_size=224)#random crop or resize
            
            if np.random.rand()>0.5:
                img = cv2.flip(img,0)
            img = img/255.0
            
            img[:,:,2] = (img[:,:,2] - 0.485)/0.229  
            img[:,:,1] = (img[:,:,1] - 0.456)/0.224
            img[:,:,0] = (img[:,:,0] - 0.406)/0.225
#            
#            img[:,:,0] = (img[:,:,0]-img[:,:,0].min())/(img[:,:,0].max()-img[:,:,0].min())
#            img[:,:,1] = (img[:,:,1]-img[:,:,1].min())/(img[:,:,1].max()-img[:,:,1].min())
#            img[:,:,2] = (img[:,:,2]-img[:,:,2].min())/(img[:,:,2].max()-img[:,:,2].min())
            img = self.transform(img)
#            img = img.float()
            
        label = self.labels[idx]
        
        return img, label
#%%