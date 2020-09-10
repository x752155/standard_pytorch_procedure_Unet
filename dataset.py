import os
from PIL import Image
import numpy as np
import cv2

route_img = r'G:/data/train/image/'
route_label = r'G:/data/train/label/'

files = os.listdir(route_label)
num =0
file_example =""
for file in files:
    print(file)
    num=num+1
    if(num==10000):
        file_example =file
print(num)

img = cv2.imread('G:/data/train/label/467.png',-1)
print(img) #尺寸为256*256*3
