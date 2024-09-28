import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# トレーニングデータセットのダウンロード
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 画像１枚あたりのスピン数
num_spin = training_data[0][0].shape[1]*training_data[0][0].shape[2]

print(num_spin)