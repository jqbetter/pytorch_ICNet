import torch
import torch.nn as nn
import numpy as np
from config import args
from model import ICNet,mean_iou


model=ICNet(args.num_classes,args.image_shape)
model=model.to(args.device)

def eval(model,data_loader,args=args):
    model.eval()


    for _,(data,label,lab_sub1,lab_sub2,lab_sub4) in enumerate(data_loader):

        _,_,logit=model(data)
        pred=torch.argmax(logit,dim=1).int()
        miou=mean_iou(pred,lab_sub1,args.num_classes)



