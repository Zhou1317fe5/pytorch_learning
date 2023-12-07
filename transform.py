"""
工具箱用于对图片进行变化
特定格式的图片->transform工具箱->结果

输入：
输出：
作用：
"""
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

#img_path = "code\\data\\hymenoptera_data\\train\\ants\\0013035.jpg"
img_path ="D:\\CodeProjects\\jupyter-notebook\\Pytorch\\code\\data\\hymenoptera_data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)
#print(img) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x25BDC447970>: PIL image类型，RGB模式，大小768x512


writer = SummaryWriter("logs")

# 常用transform使用方法
# result = transform_tool(input)

#1
# ToTensor 方法：Convert a PIL Image or ndarray to tensor。输入picture(pil.image or numpy)，输出tensor
    
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("img_tensor",img_tensor)


#2
# Compose 方法：Composes several transforms together.

#3
#ToPILImage 方法：Convert a tensor or an ndarray to PIL Image

#4
#Normalize 方法：Normalize a tensor image with mean and standard deviation.

print(img_tensor[0][0][0]) # 归一化前
trans_norm = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]) # 归一化后的像素值[-1,1]
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0]) # 归一化后
writer.add_image("Normalize",img_norm)


#5
#Resize 方法：Resize the input image to the given size.
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL ->resize ->img_resize PIL
img_resize = trans_resize(img)
print(img_resize.size)
# img PIL ->totensor ->img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)

#Compose-reszie
trans_resize_2 = transforms.Resize(225)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor]) # 前面的输出类型是后面的输入类型
img_resize_2 = trans_compose(img) # PIL img
writer.add_image("Resize",img_resize_2,1)

#6
#RandomCrop 随机裁剪方法：
trans_random = transforms.RandomCrop(512)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)



writer.close
