import torch

class Args(object):
    def __init__(self,device):
        self.device=device
        self.data_dir='./data/'       # 数据所在文件夹
        self.train_data_list="train_data_list.txt"  #训练文件列表存储文件
        self.eval_data_list="eval_data_list.txt"
        self.images_dir="./data/images/"
        self.labels_dir="./data/labels/"
        self.log_dir="./log/"                  #日志存储路径
        self.test_image_dir="./test_image/"
        self.csv_dir="./data/class_dict.csv"
        self.checkpoints="./checkpoints/miou_{}.path".format

        self.crop_height=224
        self.crop_width=224
        self.scale=(self.crop_height,self.crop_width)

        self.num_workers=0
        self.batch_size=4                       # 设置训练时的batch_size
        self.num_classes=3                     # 标签的类数

        self.image_shape =224                   # 正方形图像的高或者宽
        self.learning_rate = 0.1
        self.weight_decay=0.001
        self.momentum = 0.9
        self.base_lr=0.001                     # 初始的学习率

        self.num_epochs=20                   # 总的epochs数
        self.epoch_start_i=1
        self.is_train=True
        self.opt_params={"lr":self.learning_rate,
                         "momentum":self.momentum,
                         "weight_decay":self.weight_decay,}


args=Args(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
