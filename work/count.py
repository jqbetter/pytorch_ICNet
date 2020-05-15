from work.utils import *
from data_reader import reader_csv
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from torchvision import transforms

def draw_features(width,height,x,savename):
    fig=plt.figure(figsize=(16,16))
    #fig.subplots_adjust(left=0.05,right=0.95,bottom=0.95,wspace=0.05,hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width,i+1)
        plt.axis('off')
        img=x[0,1,:,:]
        pmin=np.min(img)
        pmax=np.max(img)
        img=(img-pmin)/(pmax-pmin+0.000001)
        plt.imshow(img,cmap='gray')
    fig.savefig(savename,dpi=100)
    fig.clf()
    plt.close()

def predict_on_image(model, epoch, csv_path, args):
    # pre-processing on image
    image = cv2.imread("demo/ceshi.png", -1)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')
    image = transforms.ToTensor()(image).unsqueeze(0)
    #read csv label path
    label_info = reader_csv(csv_path)
    # predict
    model.eval()
    predict,_,_ = model(image.cuda())
    #with torch.no_grad():
        #image1 = cv2.imread("demo/ceshi.png", -1)
        #predict = model(image.cuda())
        #predict=predict.cpu().numpy()
        #predict=predict[0,1,:,:]


        #pmin=np.min(predict)
        #pmax=np.max(predict)
        #predict=((predict-pmin)/(pmax-pmin+0.000001))*225
        #predict=predict.astype(np.uint8)
        #predict=cv2.applyColorMap(predict,cv2.COLORMAP_JET)
        #predict=predict[:,:,::-1]
        #predict = image1+predict*0.3
        #plt.imshow(predict, cmap='gray')
        #save_path = 'demo/epoch_%d.png' % (epoch)
        #cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
    w =predict.size()[-1]
    c =predict.size()[-3]
    predict = predict.resize(c,w,w)
    predict = reverse_one_hot(predict)
    predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
    predict = cv2.resize(np.uint8(predict), (224, 224))
    save_path = 'demo/epoch_%d.png' % (epoch)
    cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))

def val(model, dataloader, csv_path, epoch, loss_train_mean, writer,args):
    print('start val!')
    # label_info = get_label_info(csv_path)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 输出一张图片查看训练效果
    predict_on_image(model, epoch, csv_path, args)
    i=1
    start = time.time()
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        aTP = 0
        aFP = 0
        aTN = 0
        aFN = 0
        n00=0
        n11=0
        n22=0
        n01=0
        n02=0
        n10=0
        n12=0
        n20=0
        n21=0
        for _,(data,label,lab_sub1,lab_sub2,lab_sub4) in enumerate(dataloader):
            data = data.cuda()
            label = label.cuda()

            # get RGB predict image
            predict,_,_ = model(data)#.squeeze()
            predict.squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())
            #predict_one=predict.flatten()
            # get RGB label image
            label = label.squeeze()
            label = reverse_one_hot(label)
            label = np.array(label.cpu())
            #label_one=label.flatten()
            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist +=fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            N00, N11, N22, N01, N02, N10, N12, N20, N21 = coutpixel(label,predict)
            aFP=FP(label,predict)+aFP
            aTP=TP(label,predict)+aTP
            aFN=FN(label,predict)+aFN
            aTN=TN(label,predict)+aTN
            n00=N00+n00
            n11 = N11 + n11
            n22 = N22 + n22
            n01 = N01 + n01
            n02 = N02 + n02
            n10 = N10 + n10
            n12 = N12 + n12
            n20 = N20 + n20
            n21 = N21 + n21
            #FP=predict-label   #cloud regard as positive ,void and shadow regard as negative
            ####################cloud label:0      cloud shadow label:1           void label:2
            ####################
            #####           label
            ##             1    0
            ##predition  1 TP   FP
            ##           0 FN   TN
            # FP=label-predict
            # FP = np.where(FP > 0, 1, 0)
            # TP = FP - predict
            # FN = predict - TP
            # TN = 1 - FN - FP - TP
            # aTP = np.sum(TP) + aTP
            # aFP = np.sum(FP) + aFP
            # aTN = np.sum(TN) + aTN
            # aFN = np.sum(FN) + aFN
            #predict= torch.round(predict)
            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        dice = np.mean(precision_record)
        miou = np.mean(per_class_iu(hist))
        P = aTP / (aTP + aFP)    #precision
        R = aTP / (aTP + aFN)    #recall
        F1 = 2 * P * R / (P + R)
        Mr = (aFP + aFN) / (aTP + aTN + aFN + aFP)
        Acc = (aTP + aTN) / (aTP + aTN + aFN + aFP)
        mIOU = aTP / (aTP + aFN + aFP)
        MPA=(n00/(n00+n01+n02)+n11/(n10+n11+n12)+n22/(n20+n21+n22))/3
        end = time.time()
        writer.add_scalar('{}_dice'.format('val'), dice, epoch)
        writer.add_scalar('{}_miou'.format('val'), miou, epoch)
        writer.add_scalar('{}_P'.format('val'), P, epoch)
        writer.add_scalar('{}_R'.format('val'), R, epoch)
        writer.add_scalar('{}_F1'.format('val'), F1, epoch)
        writer.add_scalar('{}_Mr'.format('val'), Mr, epoch)
        writer.add_scalar('{}_Acc'.format('val'), Acc, epoch)
        writer.add_scalar('{}_mIOU'.format('val'), mIOU, epoch)
        #f=open("/home/ant/桌面/云影分割＋去影/results.txt",'w')
        print(n00,n11,n22,n01,n02,n10,n12,n20,n21)
        print('PA  : {:.5f}'.format(dice))  #pixcal-accuracy
        print('mIoU: {:.5f}'.format(miou))  #mean intersection over union
        print( 'P:{:.4} R:{:.4} F1:{:.4} Mr:{:.4} Acc:{:.4} mIOU:{:.4}MPA:{:.4}'.format(
             P, R, F1, Mr, Acc, mIOU,MPA))
        # f.write(str(i)+' '+str( 'P:{:.4} R:{:.4} F1:{:.4} Mr:{:.4} Acc:{:.4} mIOU:{:.4}'.format(
        #      P, R, F1, Mr, Acc, mIOU))+'\n')
        print("Time:{:.3f}s".format(end - start))
        return miou
