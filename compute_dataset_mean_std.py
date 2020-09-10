import numpy as np
import cv2
import os

# img_h, img_w = 32, 32
img_h, img_w = 256, 256   #根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []

imgs_path = 'G:/data/train/image/'
imgs_path_list = os.listdir(imgs_path)

len_ = 4000
times=2
    #len(imgs_path_list)
means_all =[]
std_all=[]
for t in range(times):
    means=[]
    stdevs=[]
    i=0
    imgs=[]

    for item in imgs_path_list[t*len_:(t+1)*len_]:
        img = cv2.imread(os.path.join(imgs_path,item))
        img = cv2.resize(img,(img_w,img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        if i%1000==0:
            print(i,'/',len_)
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    means_all.append(means)
    std_all.append(stdevs)
all =[0,0,0]
for i in range(len(means_all)):
    print(i)
    for t in range(3):
        all[t]= all[t]+means_all[i][t]
for i in range(3):
    all[i] =all[i]/times
print("all = ")
print(all)


all2=[0,0,0]
for i in range(len(std_all)):
    print(i)
    for t in range(3):
        all2[t]= all2[t]+std_all[i][t]
for i in range(3):
    all2[i] =all2[i]/times
print("all2 = ")
print(all2)
