import torch
from model_save import *
# 方式1-》保存方式1，加载模型
import torchvision


model = torch.load("../model/vgg16_method1.pth")
# print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False) # 新建网络模型结构
vgg16.load_state_dict(torch.load("./model/vgg16_method2.pth")) # 网络模型加载参数
# print(vgg16)

# 方式1陷阱
model = torch.load('./model/demo_method1.pth') # 需要先引入模型 from model_save import *
print(model)