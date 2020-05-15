import cv2
import torch
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import datetime
import csv
import pandas as pd
import numpy as np
from config import args


#随机产生训练集和测试集
def file_names(name_folder):
    fns=[]
    folder_files=os.listdir(name_folder)
    for file_name in folder_files:
        if file_name.split(".")[1] in ".png":
            fns.append(file_name)
    return fns

def data_partition(code_tset=False):   #随机选择测试集和训练集

    images_folder_files = file_names(args.images_dir)
    #labels_folder_files = file_names(agrs.labels_dir, image_format)

    np.random.shuffle(images_folder_files)
    total_num = len(images_folder_files)
    train_num = int(0.8 * total_num)
    if code_tset:
        total_num=10
        train_num=8

    with open(args.data_dir + args.train_data_list, "w") as f:
        for index in range(train_num):
            train_img = args.images_dir + images_folder_files[index]
            train_lab = args.labels_dir + images_folder_files[index]#.replace(pre_ext,last_ext)
            lines = train_img + "\t" + train_lab + "\n"
            f.write(lines)

    with open(args.data_dir + args.eval_data_list, "w") as f:
        for index in range(train_num, total_num):
            eval_img = args.images_dir + images_folder_files[index]
            eval_lab = args.labels_dir + images_folder_files[index]#.replace(pre_ext,last_ext)
            lines = eval_img + "\t" + eval_lab + "\n"
            f.write(lines)

#create and get label information
def writer_csv(csv_dir,operator="w",headers=None,lists=None):
    with open(csv_dir,operator,newline="") as csv_file:
        f_csv=csv.writer(csv_file)
        f_csv.writerow(headers)
        f_csv.writerows(lists)

def reader_csv(csv_dir):
    ann = pd.read_csv(csv_dir)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label

def scale_image(input,factor):
    #input.shape=[m,n],output.shape=[m//factor,n//factor]
    #将原tensor压缩factor

    h=input.shape[0]//factor
    w=input.shape[1]//factor

    return cv2.resize(input,(w,h),interpolation=cv2.INTER_NEAREST)

#show or save figure
def sub_2_imshow(tensor,en_save=False):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.pause(1)
    if en_save:
        now=datetime.datetime.strftime(datetime.datetime.now(),"%Y_%m_%d_%H_%M_%S")
        plt.savefig(args.test_image_dir+now+".png")
    else:
        plt.pause(3)

def sub_imshow(tensor,en_save=False):
    assert type(tensor)==torch.Tensor
    if len(tensor.shape)==4:
        for i in range(tensor.shape[0]):
            sub_2_imshow(tensor[i],en_save)
    else:
        sub_2_imshow(tensor,en_save)

def imshow(img_data,en_save=False):      #将torch.Tensor数据显示为图片
    assert type(img_data) in [torch.Tensor, tuple, list]
    if type(img_data) == torch.Tensor:
        sub_imshow(img_data,en_save)
    else:
        for img in img_data:
            sub_imshow(img,en_save)

