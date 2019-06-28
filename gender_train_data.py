
import os
import numpy as np
import tensorflow as tf

import  cv2



def get_img_list(dirname, flag=0):
    rootdir = os.path.abspath('data/' + dirname + '/')
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    files = []

    for i in range(0, len(list)):
        path = os.path.join('data/' + dirname, list[i])
        if os.path.isfile(path):
            files.append(path)
           # b=[]
            #b=cv2.imread('data\\male\\face1.bmp')
    return files


images = []
labels = []
all_images = []
all_labels=[]
a=np.array

def read_img(list,flag,):
    for i in range(len(list) - 1):
        if os.path.isfile(list[i]):
            a=cv2.imread(list[i])

            images.append(cv2.imread(list[i]).flatten())
            labels.append(flag)


#a=[]
#a=cv2.imread("data/male/files[1]").flatten()

read_img(get_img_list('male'), 0)
read_img(get_img_list('female'), 1)

images = np.array(images)
labels = np.array(labels)

# 重新打乱
permutation1=[]
permutation1=np.random.permutation(labels.shape[0])
#all_images= np.random.permutation(images)
#all_labels= np.random.permutation(labels)
all_images = images[permutation1]
all_labels = labels[permutation1]


# 训练集与测试集比例 8：2
train_total = all_images.shape[0]
train_nums = int(all_images.shape[0] * 0.8)
test_nums = all_images.shape[0] - train_nums
images = all_images[0:train_nums:]
labels = all_labels[0:train_nums:]
#b=len(images)
#lables=np.reshape(lables,(317,1))
test_images = all_images[train_nums:train_total:]
test_labels = all_labels[train_nums:train_total:]
#images_input:labels_input:



