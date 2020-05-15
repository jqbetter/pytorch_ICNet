from __future__ import absolute_import,division,print_function
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

def pad_params(kernal_size, stride):
    if type(kernal_size) == int:
        kernal_height = kernal_size
        kernal_weight = kernal_size
    elif type(kernal_size) == tuple:
        kernal_height = kernal_size[0]
        kernal_weight = kernal_size[1]

    if type(stride) == int:
        stride_height = stride
        stride_weight = stride
    elif type(stride) == tuple:
        stride_height = stride[0]
        stride_weight = stride[1]

    padding_h = max(kernal_height - stride_height, 0)
    padding_w = max(kernal_weight - stride_weight, 0)
    padding_top = padding_h // 2
    padding_left = padding_w // 2
    padding_bottom = padding_h - padding_top
    padding_right = padding_w - padding_left

    return (padding_left, padding_right, padding_top, padding_bottom)


class ICNet(nn.Module):

    def __init__(self,num_classes,input_shape):
        super(ICNet, self).__init__()
        self.num_classes=num_classes
        self.input_shape=input_shape

        # self.conv=nn.Conv2d
        # self.avg_pool=nn.AvgPool2d
        # self.max_pool=nn.MaxPool2d
        # self.interp=transforms.Resize
        # self.bn=nn.BatchNorm2d
        # self.zero_padding=nn.ZeroPad2d
        self.pad=F.pad

    def conv(self,input_data,in_channels,out_channels,kernel_size,stride,en_pad=False,dilation=1,biased=False):
        padding=0
        if en_pad:
            padding=self.pad_params(kernel_size,stride)
        tmp_op=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=biased)
        return nn.ReLU(tmp_op(input_data))

    def pool(self,input_data,kernel_size,stride,padding=0,pool_mode="avg"):
        tmp_op=nn.AvgPool2d(kernel_size,stride,padding)
        if pool_mode=="max":
            tmp_op=nn.MaxPool2d(kernel_size,stride,padding)
        return tmp_op(input_data)

    def interp(self,input_data,out_shape):
        tmp_op=transforms.Resize(out_shape)
        return tmp_op(input_data)

    def bn(self,input_data,relu=False):
        tmp_op=nn.BatchNorm2d(input_data.shape[0],momentum=0.95,eps=1e-5)
        if relu:
            return nn.ReLU(tmp_op(input_data))
        return tmp_op(input_data)

    def res_block(self,input_data,in_channels,out_channels,padding=0,dilation=None):
        tmp=self.conv(input_data,in_channels,out_channels//4,1,1)
        tmp=self.bn(tmp,True)
        tmp=self.pad(tmp,padding)
        if dilation is None:
            tmp=self.conv(tmp,tmp.shape[0],out_channels//4,3,1)
        else:
            tmp=self.conv(tmp,tmp.shape[0],out_channels//4,dilation=dilation)
        tmp=self.bn(tmp,True)
        tmp=self.conv(tmp,tmp.shape[0],out_channels,1,1)
        tmp=self.bn(tmp)
        tmp+=input_data
        return nn.ReLU(tmp)

    def proj_block(self,input_data,in_channels,out_channels,stride=1,padding=0,dilation=None):
        proj=self.conv(input_data,in_channels,out_channels,1,stride)
        proj_bn=self.bn(proj)
        tmp=self.conv(input_data,in_channels,out_channels//4,1,stride)
        tmp=self.bn(tmp,True)
        tmp=self.pad(tmp,padding)
        en_pad=padding==0
        if dilation is None:
            tmp=self.conv(tmp,tmp.shape[0],out_channels//4,3,1,en_pad)
        else:
            tmp=self.conv(tmp,tmp.shape[0],out_channels//4,3,dilation=dilation,en_pad=en_pad)
        tmp=self.bn(tmp,True)
        tmp=self.conv(tmp,tmp.shape[0],out_channels,1,1)
        tmp=self.bn(tmp)
        tmp+=proj_bn
        return nn.ReLU(tmp)

    def dilation_convs(self,input_data):
        tmp=self.res_block(input_data,input_data.shape[0],256,padding=1)
        tmp = self.res_block(tmp, tmp.shape[0], 256, padding=1)
        tmp = self.res_block(tmp, tmp.shape[0], 256, padding=1)
        tmp = self.proj_block(tmp, tmp.shape[0], 512, padding=2,dilation=2)
        tmp = self.res_block(tmp, tmp.shape[0], 512, padding=2, dilation=2)
        tmp = self.res_block(tmp, tmp.shape[0], 512, padding=2, dilation=2)
        tmp = self.res_block(tmp, tmp.shape[0], 512, padding=2, dilation=2)
        tmp = self.res_block(tmp, tmp.shape[0], 512, padding=2, dilation=2)
        tmp = self.res_block(tmp, tmp.shape[0], 512, padding=2, dilation=2)
        tmp=self.proj_block(tmp,tmp.shape[0],1024,padding=4,dilation=4)
        tmp = self.res_block(tmp, tmp.shape[0], 1024, padding=4, dilation=4)
        return self.res_block(tmp, tmp.shape[0], 1024, padding=4, dilation=4)

    def pyramis_pooling(self,input_data,input_shape):
        shape=np.ceil(input_shape//32).astype("int32")
        h,w=shape
        pool1=self.pool(input_shape,(h,w),(h,w),pool_mode="avg")
        pool1_interp=self.interp(pool1,shape)
        pool2 = self.pool(input_shape, (h//2, w//2), (h//2, w//2), pool_mode="avg")
        pool2_interp = self.interp(pool2, shape)
        pool3 = self.pool(input_shape, (h//3, w//3), (h//3, w//3), pool_mode="avg")
        pool3_interp = self.interp(pool3, shape)
        pool4 = self.pool(input_shape, (h//4, w//4), (h//4, w//4), pool_mode="avg")
        pool4_interp = self.interp(pool4, shape)
        return input_data+pool4_interp+pool3_interp+pool2_interp+pool1_interp

    def shared_convs(self,image):
        tmp=self.conv(image,image.shape[0],32,3,2,en_pad=True)
        tmp=self.bn(tmp,True)
        tmp=self.conv(tmp,tmp.shape[0],32,3,1,en_pad=True)
        tmp = self.bn(tmp, True)
        tmp = self.conv(tmp, tmp.shape[0], 32, 3, 1, en_pad=True)
        tmp = self.bn(tmp, True)
        tmp = self.pool(tmp,3,2,padding=[1,1],pool_mode="max")

        tmp = self.proj_block(tmp,tmp.shape[0], 128, padding=0)
        tmp = self.res_block(tmp,tmp.shape[0], 128, padding=1)
        tmp = self.res_block(tmp,tmp.shape[0], 128, padding=1)
        return self.proj_block(tmp,tmp.shape[0], 256, padding=1,stride=2)

    def sub_net_4(self,input_data,input_shape):
        tmp=self.interp(input_data,input_shape//32)
        tmp=self.dilation_convs(tmp)
        tmp=self.pyramis_pooling(tmp,input_shape)
        tmp=self.conv(tmp,tmp.shape[0],256,1,1)
        tmp = self.bn(tmp, relu=True)
        return self.interp(tmp, out_shape=np.ceil(input_shape / 16))

    def sub_net_2(self,input):
        return self.bn(self.conv(input,input.shape[0],128,1,1))

    def sub_net_1(self,input):
        tmp = self.conv(input, input.shape[0],32, 3, 2, en_pad=True)
        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, tmp.shape[0],32, 3, 2, en_pad=True)
        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, tmp.shape[0],64, 3, 2, en_pad=True)
        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, tmp.shape[0],128, 1, 1,en_pad=True)
        return self.bn(tmp, relu=False)

    def CCF24(self, sub2_out, sub4_out, input_shape):
        tmp = self.pad(sub4_out, padding=2)
        tmp = self.conv(tmp,tmp.shape[0],128, 3,dilation=2)
        tmp = self.bn(tmp, relu=False)
        tmp = tmp + sub2_out
        tmp = nn.ReLU(tmp)
        tmp = self.interp(tmp, input_shape // 8)
        return tmp

    def CCF124(self, sub1_out, sub24_out, input_shape):
        tmp = self.pad(sub24_out, padding=2)
        tmp = self.conv(tmp,tmp.shape[0],128, 3,dilation=2)
        tmp = self.bn(tmp, relu=False)
        tmp = tmp + sub1_out
        tmp = nn.ReLU(tmp)
        tmp = self.interp(tmp, input_shape // 4)
        return tmp

    def forward(self, data):
        image_sub1 = data
        image_sub2 = self.interp(data, out_shape=self.input_shape * 0.5)

        s_convs = self.shared_convs(image_sub2)
        sub4_out = self.sub_net_4(s_convs, self.input_shape)
        sub2_out = self.sub_net_2(s_convs)
        sub1_out = self.sub_net_1(image_sub1)

        sub24_out = self.CCF24(sub2_out, sub4_out, self.input_shape)
        sub124_out = self.CCF124(sub1_out, sub24_out, self.input_shape)
        conv6_cls = self.conv(
            sub124_out, sub124_out.shape[0], self.num_classes, 1, 1, biased=True)
        sub4_out = self.conv(
            sub4_out, sub4_out.shape[0], self.num_classes, 1, 1, biased=True)
        sub24_out = self.conv(
            sub24_out, sub24_out.shape[0], self.num_classes, 1, 1, biased=True)
        return sub4_out, sub24_out, conv6_cls



