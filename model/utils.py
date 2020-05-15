import torch
import numpy as np
import cv2
import PIL.Image

def PILImageToCv(image):
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

def CVImageToPIL(iamge):
    return PIL.Image.fromarray(cv2.cvtColor(iamge,cv2.COLOR_BGR2RGB))

def reverse_one_hot(image):
    if len(image.shape)==4:
        image=image.permute(0,2,3,1)
    elif len(image.shape)==3:
        image=image.permute(1,2,0)
    return torch.argmax(image,dim=-1)
