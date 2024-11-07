import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader

mean = 0.2860347330570221
std = 0.3530242443084717

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize(mean, std)
    ])
)

loader = DataLoader(train_set, batch_size=1)
image, label = next(iter(loader))

print(image.shape)

