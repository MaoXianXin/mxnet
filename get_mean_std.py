import torch
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

# 设置随机数种子，使结果可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

# 数据读取
trainset = torchvision.datasets.ImageFolder('/home/mao/PycharmProjects/convolution-network-natural-scenes/dataset/natural-scenes/seg_train',
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                            ]))
testset = torchvision.datasets.ImageFolder('/home/mao/PycharmProjects/convolution-network-natural-scenes/dataset/natural-scenes/seg_test',
                                           transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                           ]))

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

print(get_mean_and_std(trainset))
print(get_mean_and_std(testset))