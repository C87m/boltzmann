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

# データの加工
edit_data_list = []
for n in range(60000):
    raw_data = training_data[n][0].squeeze()
    edit_data = [1 if raw_data >= 4 else 0 for spin in range(num_spin)]
    edit_data_list.append(edit_data)