#调用官方库及第三方库
import torch
import numpy as np
from torch import nn,optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

#基础功能
from data_reader import data_partition       #选择训练集和测试集
from work.count import poly_lr_scheduler
from config import args                        #基础参数初始化

#数据读取，模型导入
from data_reader import CVReader
from data_reader import PILReader
from data_reader import reader_csv,imshow
from model import ICNet,loss
from work.count import *
from work.train import train

#代码运行预处理
data_partition()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer=SummaryWriter(log_dir=args.log_dir)

train_data=CVReader(args.data_dir+args.train_data_list)

#eval_data=CVReader(args.labels_dir+args.train_data_list,transform)

#image,label,label_sub1,label_sub2,label_sub4=CVReader(args.data_dir+args.train_data_list,transform)[0]

dataloader_train=DataLoader(dataset=train_data,batch_size=args.batch_size,num_workers=args.num_workers)

#模型选择
model=ICNet(args.num_classes,args.image_shape)

model=model.to(device)

optimizer=optim.SGD(model.parameters(),**args.opt_params)

train(model,optimizer,dataloader_train,device,args)





