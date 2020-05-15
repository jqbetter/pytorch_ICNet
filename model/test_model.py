import torch
from torch import nn
from torchvision import models

class ICNet(nn.Module):
    def __init__(self):
        super(ICNet, self).__init__()
        self.conv=nn.Conv2d(3,1,1)
        self.pool=nn.MaxPool2d(1,1)

    def forward(self,x):
        out=self.conv(x)
        out=self.pool(out)
        out=nn.Sigmoid(out)
        return out

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = None
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            features=x

        return features