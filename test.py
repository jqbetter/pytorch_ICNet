import torch
import numpy as np
from data_reader import reader_csv
from config import args


def Iou(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]])#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]])#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)#同上
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上
    batchMious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            iou = intersection / union
            ious.append(iou)
        miou = np.mean(ious)#计算该图像的miou
        batchMious.append(miou)
    return batchMious

# x=torch.zeros(3,32,32)
# y=torch.ones(3,32,32)
# classnum=2
#
# print(Iou(x,y,classnum))
#label_values=reader_csv(args.csv_dir)


