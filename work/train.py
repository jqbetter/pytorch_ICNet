#调用官方库及第三方库
import torch
import numpy as np
from torch import nn,optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datetime

#基础功能
from data_reader import data_partition       #选择训练集和测试集
from work.count import poly_lr_scheduler
from config import args                        #基础参数初始化

#数据读取，模型导入
from data_reader import CVReader
from data_reader import PILReader
from data_reader import reader_csv
from model import ICNet,loss,mean_iou
from work.count import *



LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 0.44


def train(model,optimizer,dataloader_train,device,args=args):
    model.train()
    step=0
    mIoU_cache=0.5
    mIou=0.5
    lr=args.learning_rate

    writer=SummaryWriter(log_dir=args.log_dir)

    for epoch in range(args.num_epochs):
        now=datetime.datetime.now()
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        # epoch = epoch + args.epoch_start_i
        # epoch = epoch + 1
        model.train()
        loss_record=[]

        for _,(data,label,lab_sub1,lab_sub2,lab_sub4) in enumerate(dataloader_train):

            data,label=data.to(device),label.to(device)
            lab_sub1,lab_sub2,lab_sub4=lab_sub1.to(device),lab_sub2.to(device),lab_sub4.to(device)

            #print("label",label.shape)
            #print("label_sub1",lab_sub1.shape)
            #print(lab_sub1)
            #print("labe2_sub1",lab_sub2.shape)
            #print(lab_sub2)
            #print("labe4_sub1",lab_sub4.shape)
            #print("lab_sub4",lab_sub4)

            sub4_out,sub24_out,sub124_out=model(data)
            # print("sub4_out", sub4_out.shape)
            # print("sub24_out", sub24_out.shape)
            # print("sub124_out", sub124_out)
            loss_sub4 = loss(sub4_out, lab_sub4, args.num_classes)
            loss_sub24 = loss(sub24_out, lab_sub2, args.num_classes)
            loss_sub124 = loss(sub124_out, lab_sub1, args.num_classes)
            #
            reduced_loss = LAMBDA1 * loss_sub4 + LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124
            optimizer.zero_grad()  # 梯度清零
            reduced_loss.backward()  # 计算梯度
            optimizer.step()
            step+=1
            loss_record.append(reduced_loss.item())

            # mIou = mean_iou(pred=sub124_out, label=lab_sub1, num_classes=args.num_classes)
            # print("MIOU:", mIou)
            # print("time:",datetime.datetime.now()-now)

        loss_tm=np.mean(loss_record)
        print("epoch:",epoch,"loss:",loss_tm)
        writer.add_scalar("train_loss",loss_tm,epoch)

        if (epoch+1)%5==0 and epoch!=0:
            torch.save(model.state_dict(),args.checkpoints(epoch+1))
        if (epoch+1) % 5 ==0 and epoch!=0:
            # mIou=val(model,dataloader_train,args.csv_dir,epoch,loss_tm,writer,args)
            # print("MIOU:",mIou)
            mIou = mean_iou(pred=sub124_out, label=lab_sub1, num_classes=args.num_classes)
            print("MIOU:", mIou)

        if mIou>mIoU_cache:
            torch.save(model.state_dict(),args.checkpoints(mIou))
            mIoU_cache=mIou
        print("time:", datetime.datetime.now() - now)
    torch.save(model.state_dict(),args.checkpoints("last"))
    writer.close()

#print(args.checkpoints(10.0))

