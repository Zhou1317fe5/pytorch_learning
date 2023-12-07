from torch.utils.data import Dataset
from PIL import Image
import os
class MyData(Dataset):
    def __init__(self,root_dir,label_dir): # 初始化类 提供全局变量
        self.root_dir = root_dir  # self.相当于指定类中的全局变量
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir) # 将两个地址拼接起来，拼接符号根据系统（win linux）自动选择
        self.img_path = os.listdir(self.path)


    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)

root_dir = "code\\data\\hymenoptera_data\\train" # code\data\hymenoptera_data\train
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

#img,label = ants_dataset[0]
#img.show()
#print(label)

train_dataset = ants_dataset+bees_dataset # 可用于将仿造的数据集和原数据集合并



