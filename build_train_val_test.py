import random
import os
random.seed(12345)
from config import *

L1 = random.sample(range(1, total_size), test_val_size)
print(L1)
print(len(L1))
length = test_val_size/2
test_set = L1[0:(test_val_size/2).__int__()]      #测试集合
print(test_set)
print(len(test_set))
val_set = L1[(test_val_size/2).__int__():total_size]   #交叉验证集
print(val_set)
print(len(val_set))

#================================设置txt训练集，验证机，测试集
# 先创建文件夹，如果存在的话，先删除，再创建
train_txt = open(txt_route+"train_set.txt","x")
test_txt = open(txt_route+"test_set.txt","x")
val_txt = open(txt_route+"val_set.txt","x")

files = os.listdir(route_img)
num=1
for file in files:
    if(num in test_set):
        test_txt.write(route_img+file+'\n')
    elif (num in val_set):
        val_txt.write(route_img+file+'\n')
    else:
        train_txt.write(route_img+file+'\n')
    num=num+1

print(num)
