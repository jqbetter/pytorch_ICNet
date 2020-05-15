from torch.utils.data import Dataset
import cv2
from data_reader.utils import scale_image
from torchvision import transforms

class CVReader(Dataset):

    def __init__(self,data_dir):
        super(CVReader, self).__init__()

        self.data_dir=data_dir
        self.to_tensor = transforms.ToTensor()

        self.image_list=[]
        self.label_list=[]
        with open(self.data_dir,"r") as f:
            for line in f.readlines():
                img,lab=line.strip().split()
                self.image_list.append(img.strip())
                self.label_list.append(lab.strip())

    def __getitem__(self, idx):
        image=cv2.imread(self.image_list[idx])
        label=cv2.imread(self.label_list[idx])
        #print("reader image",image)
        # print("cv2readimage.max", image.max())
        # print("cv2readlabel.max", label.max())

        label_sub1=scale_image(label,4)
        label_sub2=scale_image(label,8)
        label_sub4=scale_image(label,16)
        #label=scale_image(label,1)


        image=self.to_tensor(image)
        label=self.to_tensor(label)
        # label_sub1 = self.to_tensor(label_sub1)
        # label_sub2 = self.to_tensor(label_sub2)
        # label_sub4 = self.to_tensor(label_sub4)

        return image,label,label_sub1,label_sub2,label_sub4

    def __len__(self):
        return len(self.image_list)
