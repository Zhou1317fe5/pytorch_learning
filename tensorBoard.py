from torch.utils.tensorboard import SummaryWriter # pip install tensorboard
import cv2
import numpy as np
from PIL import  Image
# 1 初始化函数创建实例，输入文件夹名称（可选），用于存储事件文件
writer = SummaryWriter("logs")

# 2 创建两个方法 -添加img -添加scalar
# writer.add_image() # 写图片
#writer.add_scalar() # 写数据
'''
for i in range(100):
    writer.add_scalar("y=x", i, i) # 参数1 表头 ，参数2 scalar_value 数值 (相当于y轴)，参数3 global_step 训练步数 (相当于x轴)
writer.close()
'''


# 3 打开logs中的事件文件
# Terminal中：tensorboard --logdir=事件文件所在的地址(无数据用绝对地址) --port=端口号  (端口号不指定默认6006)
# tensorboard --logdir=logs --port=6007




img_path = "data/hymenoptera_data2/train/ants_image/7759525_1363d24e88.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(img_array.shape) # output:(512, 768, 3) HWC(高度 宽度 通道数) 形状

writer.add_image("train", img_array, 1, dataformats='HWC') # 默认CHW，需用dataformats指定与图片相同的形式
writer.close()