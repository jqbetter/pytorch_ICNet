from __future__ import absolute_import,division,print_function
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F


def pad_params(kernal_size, stride):   #padding paramters (l,r,t,b)
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

class interp(nn.Module):
    def __init__(self,size=None,scale=None):
        super(interp, self).__init__()
        self.size=size
        self.scale=scale
        if self.scale != None:
            self.size=None

    def forward(self,input):
        #return nn.UpsamplingBilinear2d(self.size)(input)
        return F.interpolate(input=input,size=self.size,scale_factor=self.scale,mode="bilinear")

class conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel=1,stride=1,en_pad=False,dilation=1,biased=False,padding=0):
        super(conv, self).__init__()
        self.en_pad=en_pad
        self.kernel=kernel
        self.stride=stride
        self.padding=padding
        self.conv=nn.Conv2d(in_channels,out_channels,kernel,stride,dilation=dilation,bias=biased)

    def forward(self,input):
        tmp=input
        if self.en_pad:
            tmp=F.pad(tmp,pad_params(self.kernel,self.stride),value=self.padding)
        return self.conv(tmp)

class pool(nn.Module):
    def __init__(self,kernel,stride=1,padding=0,pool_mode="avg"):
        self.pool=nn.AvgPool2d(kernel,stride,padding)
        if pool_mode=="max":
            self.pool=nn.MaxPool2d(kernel,stride,padding)
    def forward(self,input):
        return self.pool(input)

class pyramis_pooling(nn.Module):
    #out_channels=2*in_channels
    def __init__(self,in_channels,input_shape):  #input_shape=image.shape[1:]
        super(pyramis_pooling, self).__init__()
        self.in_channels=in_channels
        self.shape=input_shape//32

        self.pool_1 = nn.AdaptiveAvgPool2d(self.shape)
        self.pool_2 = nn.AdaptiveAvgPool2d(self.shape//2)
        self.pool_3 = nn.AdaptiveAvgPool2d(self.shape//3)
        self.pool_4 = nn.AdaptiveAvgPool2d(self.shape//4)
        self.conv_reduce=nn.Conv2d(self.in_channels,self.in_channels//4,1,1)
        self.iterp = interp(size=self.shape)

    def forward(self,input):
        csp=input
        cs1 = self.iterp(self.conv_reduce((self.pool_1(csp))))
        cs2 = self.iterp(self.conv_reduce((self.pool_2(csp))))
        cs3 = self.iterp(self.conv_reduce((self.pool_3(csp))))
        cs4 = self.iterp(self.conv_reduce(self.pool_4(csp)))
        return torch.cat((csp, cs1, cs2, cs3, cs4), dim=1)

class res_block(nn.Module):
    def __init__(self,in_channels,out_channels,padding=0,dilation=None):
        super(res_block, self).__init__()
        self.in_c=in_channels
        self.out_c=out_channels
        self.padding=padding
        self.dilation=dilation

        self.conv_1=nn.Conv2d(self.in_c,self.out_c//4,1,1)
        self.bn_1 = nn.BatchNorm2d(self.out_c//4, eps=1e-5, momentum=0.95)
        self.relu_1=nn.ReLU()
        self.zero_pad=F.pad
        self.conv_2 = nn.Conv2d(self.out_c // 4, self.out_c // 4, 3, 1)
        if dilation != None:
            self.conv_2 = nn.Conv2d(self.in_c // 4, self.out_c // 4, 3,dilation=self.dilation)
        self.bn_2= nn.BatchNorm2d(self.out_c//4, eps=1e-5, momentum=0.95)
        self.relu_2 = nn.ReLU()
        self.conv_3=nn.Conv2d(self.out_c//4,self.out_c,1,1)
        self.bn_3=nn.BatchNorm2d(self.out_c, eps=1e-5, momentum=0.95)
        self.relu_3=nn.ReLU()

    def forward(self,input):
        tmp=input
        tmp=self.relu_1(self.bn_1(self.conv_1(tmp)))
        tmp=self.zero_pad(tmp,(self.padding,self.padding,self.padding,self.padding))
        tmp = self.relu_2(self.bn_2(self.conv_2(tmp)))
        tmp=self.bn_3(self.conv_3(tmp))
        return self.relu_3(tmp+input)

class proj_block(nn.Module):
    def __init__(self,in_channels,out_channels,padding=0,dilation=None,stride=1):
        super(proj_block, self).__init__()
        self.in_c=in_channels
        self.out_c=out_channels
        self.padding=padding
        self.dilation=dilation
        self.stride=stride
        self.scale_ratio=3

        self.conv_1=nn.Conv2d(self.in_c,self.out_c,1,self.stride)
        self.bn_1=nn.BatchNorm2d(self.out_c, eps=1e-5, momentum=0.95)
        self.conv_2=nn.Conv2d(self.out_c,self.out_c//self.scale_ratio,1,self.stride)
        self.bn_2=nn.BatchNorm2d(self.out_c//self.scale_ratio, eps=1e-5, momentum=0.95)
        self.relu_2=nn.ReLU()
        self.zero_pad=F.pad
        self.en_pad=False
        if self.padding==0:
            self.en_pad=True
        self.conv_3=conv(self.out_c//self.scale_ratio,self.out_c//self.scale_ratio,3,1,en_pad=self.en_pad)
        if self.dilation != None:
            self.conv_3=conv(self.out_c//self.scale_ratio,self.out_c//self.scale_ratio,3,1,self.en_pad,dilation)
        self.bn_3=nn.BatchNorm2d(self.out_c//self.scale_ratio, eps=1e-5, momentum=0.95)
        self.relu_3=nn.ReLU()
        self.conv_4=nn.Conv2d(self.out_c//self.scale_ratio,self.out_c,1,1)
        self.bn_4=nn.BatchNorm2d(self.out_c, eps=1e-5, momentum=0.95)
        self.relu_4=nn.ReLU()

    def forward(self,input):
        proj_bn=self.bn_1(self.conv_1(input))
        tmp=self.relu_2(self.bn_2(self.conv_2(proj_bn)))
        tmp=self.zero_pad(tmp,(self.padding,self.padding,self.padding,self.padding))
        tmp=self.relu_3(self.bn_3(self.conv_3(tmp)))
        tmp=self.bn_4(self.conv_4(tmp))
        SIZE=tmp.shape[2]
        if proj_bn.shape[2]>tmp.shape[2]:
            SIZE=proj_bn.shape[2]
            tmp=interp(size=SIZE)(tmp)
            return self.bn_4(tmp+proj_bn)
        proj_bn=interp(size=SIZE)(proj_bn)
        return self.bn_4(tmp+proj_bn)


class dilation_convs(nn.Module):
    #out_channels=1024
    def __init__(self,in_channels):
        super(dilation_convs, self).__init__()
        self.in_c=in_channels
        self.res_block_1 = res_block(self.in_c,256,1)
        self.res_block_2 = res_block(256, 256, 1)
        self.res_block_3 = res_block(256, 256, 1)
        self.proj_block_1=proj_block(256,512,2,2)
        self.res_block_4 = res_block(512,512,2,2)
        self.res_block_5 = res_block(512, 512, 2, 2)
        self.res_block_6 = res_block(512, 512, 2, 2)
        self.res_block_7 = res_block(512, 512, 2, 2)
        self.res_block_8 = res_block(512, 512, 2, 2)
        self.proj_block_2=proj_block(512,1024,4,4)
        self.res_block_9=res_block(1024,1024,4,4)
        self.res_block_10 = res_block(1024, 1024, 4, 4)

    def forward(self,input):
        tmp=self.res_block_3(self.res_block_2(self.res_block_1(input)))
        tmp=self.proj_block_1(tmp)
        tmp=self.res_block_8(self.res_block_7(self.res_block_6(self.res_block_5(self.res_block_4(tmp)))))
        tmp=self.proj_block_2(tmp)
        return self.res_block_10(self.res_block_9(tmp))

class shared_convs(nn.Module):
    #out_channels=256
    def __init__(self,in_channels):
        super(shared_convs, self).__init__()
        self.conv_1=conv(in_channels,32,3,2,en_pad=True)
        self.bn_1=nn.BatchNorm2d(32,momentum=0.95)
        self.relu_1=nn.ReLU()
        self.conv_2=conv(32,32,3,1,en_pad=True)
        self.bn_2=nn.BatchNorm2d(32,momentum=0.95)
        self.relu_2=nn.ReLU()
        self.conv_3=conv(32,64,3,1,en_pad=True)
        self.bn_3=nn.BatchNorm2d(64,momentum=0.95)
        self.relu_3=nn.ReLU()
        self.max_pool=nn.MaxPool2d(kernel_size=3,stride=2,padding=[1,1])
        self.proj_block_1=proj_block(64,128)
        self.res_block_1=res_block(128,128,1)
        self.res_block_2=res_block(128,128,1)
        self.proj_block_2=proj_block(128,256,1,stride=2)

    def forward(self,input):
        tmp=self.relu_1(self.bn_1(self.conv_1(input)))
        tmp=self.relu_2(self.bn_2(self.conv_2(tmp)))
        tmp=self.relu_3(self.bn_3(self.conv_3(tmp)))
        tmp=self.max_pool(tmp)
        tmp=self.res_block_1(self.proj_block_1(tmp))
        tmp=self.res_block_2(tmp)
        return self.proj_block_2(tmp)

class sub_net_1(nn.Module):
    def __init__(self):
        super(sub_net_1, self).__init__()
        self.conv_1=conv(3,32,3,2,True)
        self.bn_1=nn.BatchNorm2d(32,eps=1e-5,momentum=0.95)
        self.conv_2=conv(32,32,3,2,True)
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.95)
        self.conv_3=conv(32,64,3,2,True)
        self.bn_3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.95)
        self.conv_4=conv(64,128,1,1)
        self.bn_4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.95)
        self.relu=nn.ReLU()

    def forward(self,input):
        tmp = self.relu(self.bn_1(self.conv_1(input)))
        tmp = self.relu(self.bn_2(self.conv_2(tmp)))
        tmp = self.relu(self.bn_3(self.conv_3(tmp)))
        tmp = self.bn_4(self.conv_4(tmp))
        return tmp

class sub_net_2(nn.Module):
    #out_channels=128
    def __init__(self,in_channels):
        super(sub_net_2, self).__init__()
        self.conv=conv(in_channels,128,1,1)
        self.bn=nn.BatchNorm2d(128,momentum=0.95)

    def forward(self,input):
        return self.bn(self.conv(input))

class sub_net_4(nn.Module):
    def __init__(self,in_channels,input_shape):
        super(sub_net_4, self).__init__()
        self.interp_1=interp(input_shape//32)
        self.dilation_conv=dilation_convs(in_channels)
        self.psp_net=pyramis_pooling(1024,input_shape)
        self.conv_1=nn.Conv2d(2048,256,1,1)
        self.bn_1=nn.BatchNorm2d(256,momentum=0.95)
        self.relu_1=nn.ReLU()
        self.interp_2=interp(input_shape//16)

    def forward(self,input):
        tmp=self.interp_1(input)
        tmp=self.dilation_conv(tmp)
        tmp=self.psp_net(tmp)
        tmp=self.relu_1(self.bn_1(self.conv_1(tmp)))
        return self.interp_2(tmp)

class CFF(nn.Module):
    #CFF124:scale=4,CFF24:scale=8
    def __init__(self,in_channels,input_shape,scale):
        super(CFF,self).__init__()
        self.zero_padding=nn.ZeroPad2d(padding=2)
        self.atrous_conv=nn.Conv2d(in_channels,128,3,dilation=2)
        self.bn=nn.BatchNorm2d(128,momentum=0.95)
        self.relu=nn.ReLU()
        self.interp=interp(size=input_shape//scale)

    def forward(self,input_1,input_2):
        tmp=self.zero_padding(input_2)
        tmp=self.atrous_conv(tmp)
        tmp=self.bn(tmp)
        # SIZE=tmp.shape[2]
        # if input_1.shape[2]>tmp.shape[2]:
        #     SIZE=input_1.shape[2]
        #     tmp=interp(size=SIZE)(tmp)
        #     return self.interp(self.relu(input_1+tmp))
        # input_1=interp(size=SIZE)(input_1)
        tmp=self.relu(input_1+tmp)
        return self.interp(tmp)

class ICNet(nn.Module):

    def __init__(self,num_classes,input_shape):
        super(ICNet, self).__init__()
        self.num_classes=num_classes
        self.input_shape=input_shape

        self.interp_1=interp(size=self.input_shape//2)  #input data:orgain image,output data:image_sun2
        self.sconv=shared_convs(3)      #input data:image_sub2,output data:s_convs
        self.snet_4=sub_net_4(256,self.input_shape)  #input data:s_convs,output data:sub4_out
        self.snet_2=sub_net_2(256)      #input data:s_convs,output data:sub2_out
        self.snet_1=sub_net_1()         #input data:origain image,output data:sub1_out

        self.cff24=CFF(256,self.input_shape,8)  #input data:sub2_out and sub4_out,output data:sub24_out
        self.cff124=CFF(128,self.input_shape,4)  #input data:sub1_out and sub24_out,output data:sub124_out

        self.conv_cls_conv=nn.Conv2d(128,self.num_classes,1,1)  #input data:sun124_out,output data:conv6_cls
        self.sub4_out_conv=nn.Conv2d(256,self.num_classes,1,1)  #input data:sub4_out,output data:sub4_out
        self.sub24_out_conv=nn.Conv2d(128,self.num_classes,1,1)  #input data:sub24_out,output data:sub24_out

    def forward(self,input):
        image_sub1=input
        image_sub2=self.interp_1(input)

        s_convs=self.sconv(image_sub2)
        sub4_out=self.snet_4(s_convs)
        sub2_out=self.snet_2(s_convs)
        sub1_out=self.snet_1(image_sub1)

        sub24_out=self.cff24(sub2_out,sub4_out)

        sub124_out=self.cff124(sub1_out,sub24_out)

        conv_cls=self.conv_cls_conv(sub124_out)
        sub4_out=self.sub4_out_conv(sub4_out)
        sub24_out=self.sub24_out_conv(sub24_out)

        return sub4_out,sub24_out,conv_cls