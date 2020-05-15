from torch.utils.data import Dataset
from PIL import Image
from data_reader.utils import reader_csv
from torchvision import transforms
from config import args

class PILReader(Dataset):

    def __init__(self,data_dir,):
        super(PILReader, self).__init__()

        self.data_dir=data_dir
        self.label_info=reader_csv(args.csv_dir)
        self.to_tensor = transforms.ToTensor()
        self.resize_image=transforms.Resize(args.scale,Image.BILINEAR)
        self.resize_label=transforms.Resize(args.scale,Image.NEAREST)
        self.resize = transforms.Resize

        self.image_list=[]
        self.label_list=[]
        with open(self.data_dir,"r") as f:
            for line in f.readlines():
                img,lab=line.strip().split()
                self.image_list.append(img.strip())
                self.label_list.append(lab.strip())
        try:
            assert (len(self.image_list)==len(self.label_list))
        except AssertionError as AE:
            print("Sample and Label not match!")

    def __getitem__(self, idx):
        image=Image.open(self.image_list[idx]).convert("RGB")
        label=Image.open(self.label_list[idx]).convert("RGB")


        image=self.to_tensor(image)
        label=self.to_tensor(label)

        return image,label

    def __len__(self):
        return len(self.image_list)

