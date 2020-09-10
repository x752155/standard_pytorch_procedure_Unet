#Dataset类是Pytorch中图像数据集中最为重要的一个类，也是Pytorch中所有数据集加载类中应该继承的父类。
# 其中父类中的两个私有成员函数必须被重载，len()与get_item()
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import cv2
import config
from loss import *


class MyDataset(Dataset):
    def __init__(self,train_set_txt,tansform =None,is_test=False):
        super(MyDataset,self).__init__()
        train_file = open(train_set_txt,'r')
        imgs = []
        for line in train_file:
            line = line.strip('\n')
            imgs.append(line)
        self.imgs =imgs
        if tansform ==None:
            self.transform = transforms.Compose([
            #transforms.Resize(config.resize_scale),
            #transforms.RandomResizedCrop(config.resize_scale[0]),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.img_mean, std=config.img_std)
        ])
            self.is_test = is_test
    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index])

        (filepath, tempfilename) = os.path.split(self.imgs[index])
        (filename, extension) = os.path.splitext(tempfilename)
        #print(config.route_label+filename+'.png')
        label = cv2.imread(config.route_label+filename+'.png',-1) #16位单通道必须这么读
        label = label/100       #百位数转化为个位数
        label =label [:, :,np.newaxis] # 向后增加一个维度
        label = mask_to_onehot(label,[[1],[2],[3],[4],[5],[6],[7],[8]])
        label = torch.from_numpy(label)
        label = label.permute(2,1,0)

        if  self.is_test== False:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return  len(self.imgs)


