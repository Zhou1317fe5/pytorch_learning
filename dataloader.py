import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor())


# DataLoader常用参数：
# dataset:dataset数据
# batch_size(int):每次抓取batch_size个样本
# shuffle(bool):是否打乱
# num_workers(int):抓取时是否使用多进程，windows系统中使用多进程可能会出bug，改为0即可
# drop_last(bool):最后无法整取的样本是否丢弃


# batch_size

"""
当dataloader(batch_size=4)时：

img0,target0=dataset[0]
img1,target1=dataset[1]
img2,target2=dataset[2]
img3,target3=dataset[3]

imgs = [img0,img1,img2,img3]
targets = [target0,target1,target2,target3]
"""

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True,num_workers=0,drop_last=False)
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs,targets = data
    writer.add_images("batch_size=4",imgs,step)
    #writer.add_images("batch_size=64_drop_last=False",imgs,step)
    #writer.add_images("batch_size=64_drop_last=True",imgs,step)
    #writer.add_images("batch_size=64_shuffle=True",imgs,step)
    step += 1

writer.close()