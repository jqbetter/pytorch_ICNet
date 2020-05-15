import numpy as np
import torch
from torch import nn
from config import args
import cv2
from torchvision import transforms
from model.utils import PILImageToCv,CVImageToPIL,reverse_one_hot
from data_reader.utils import imshow
from datetime import datetime


def loss(logit,label,num_classes):
    #logit.shape=[N2,num_classes,h2,w2],label.shape=[N1,3,h1,w1]
    num_classes_tensor=torch.full(label.shape,num_classes).cuda()
    #num_classes_tensor=num_classes_tensor.permute([0, 3, 1, 2])

    label_nignore=label<num_classes_tensor.byte()
    label_nignore=label_nignore.float()

    #logit=logit.permute([0,2,3,1])   #改变logit的维度顺序如org.shape=[1,2,3,4]，则dst.shape=[1,3,4,2]
    #logit=logit.contiguous()     #重新开辟内存空间存储数据

    logit=logit.reshape([-1,num_classes])
    row,_ = logit.shape
    label_nignore = label_nignore.reshape([-1, 1])
    logit = nn.Softmax(dim=1)(logit)

    label=label.reshape([row,-1])
    _,column = label.shape

    loss=0.
    for i in range(column):
        label_i=label[:,i]
        label_i = label_i.clamp(0, num_classes).int()
        loss+=nn.CrossEntropyLoss(ignore_index=num_classes)(logit,label_i.long())#torch.tensor([1024],dtype=torch.float32))

    loss/=column

    area = torch.mean(label_nignore)
    area=max(0.1,area)
    return loss/area

#count miou
def iou(x,y,iou=True):
    assert type(y) == list and type(x)==list
    union = set(x).union(y)
    insec = set(x).intersection(y)

    union = [value for _, value in enumerate(union)]
    insec = [value for _, value in enumerate(insec)]
    if iou:
        return len(insec) / max(len(union),1)
    return union,insec

def mean_iou(pred,label,num_classes=args.num_classes):
    #pre_time=datetime.now()
    #print("now time",pre_time)
    # imshow(pred,True)
    #image = pred.cpu().clone()
    tmps_p=[]
    for tmp_p in pred:
        #sub = ele.squeeze(0)
        # sub = transforms.ToPILImage()(ele)
        # sub=PILImageToCv(sub)
        sub_p = tmp_p.permute(1, 2, 0)
        #print("sub",sub)
        sub_p=np.array(torch.argmax(sub_p,dim=-1).cpu())
        #print("sub after argmax", sub)
        # print("type（sub_p）",type(sub_p),sub_p.shape)
        # print("sub_p",sub_p)
        #print("sub.shape",sub.shape,sub.max(),type(sub))
        tmps_p.append(sub_p)

    tmps_l = []
    for tmp_l in label:
        sub_l = tmp_l.permute(1, 2, 0)
        sub_l = np.array(torch.argmax(tmp_l, dim=-1).cpu())
        # print("type（sub_l）", type(sub_l), sub_l.shape)
        # print("sub_l", sub_l)
        tmps_l.append(sub_l)

    # print(tmps_p)
    # print("====================================")
    # print(tmps_l)
    pred=torch.Tensor(tmps_p).to(args.device).byte()
    label=torch.Tensor(tmps_l).to(args.device).byte()
    #print("pred:",pred)
    #print("label:",label)
    # pred=pred.byte()
    # print("pred paramters:",pred.shape,pred.max(),pred.min(),"\n",pred)
    # print("label paramters:",label.shape,label.max(),label.min(),"\n",label)
    #print("use time:",datetime.now()-pre_time)
    # print(pred)
    # print(label)
    #print("loss temp",pred.shape)
    #print("loss pred",pp.shape,pp.max())
    #print("loss label", label.shape)
    l_min=label.min()
    label=label.clamp(l_min,args.num_classes)
    nc_tensor=torch.full(label.shape,fill_value=num_classes-1,device=args.device).byte()#,device=device)
    label_ignore=label==nc_tensor
    label_nignore=label!=nc_tensor
    #pred=pred.permute([0,2,3,1])
    #print("after permute shape",pred.shape)


    # print("pred",pred.shape)
    # print("label_nignore",label_nignore.shape)
    pred=pred*label_nignore+label_ignore*(args.num_classes-1)
    #print("after process",pred)

    pred,label=pred.flatten(),label.flatten()
    # print("label",label)
    # print("pred",pred)

    label_locals={}
    for classes in range(num_classes):
        label_locals[str(classes)]=[]
        local=0
        for value in label:
            if value == classes:
                label_locals[str(classes)].append(local)
            local+=1

    pred_locals = {}
    for classes in range(num_classes):
        pred_locals[str(classes)]=[]
        local=0
        for value in pred:
            if value == classes:
                pred_locals[str(classes)].append(local)
            local+=1

    ious=0.
    for classes in range(num_classes):
        ious+=iou(label_locals[str(classes)],pred_locals[str(classes)],True)

    return ious/num_classes



