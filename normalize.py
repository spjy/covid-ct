import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
import os
import numpy as np
import matplotlib.pyplot as plt
import math

# TODO: Construct your data in the following baseline structure: 1) ./Dataset/Train/image/, 2) ./Dataset/Train/label, 3) ./Dataset/Test/image, and 4) ./Dataset/Test/label
class LungDataset(Dataset):
    def __init__(self, root, transform):        
        self.root = root
        self.transform = transform

    def __len__(self):
        # Return number of points in the dataset
        imgs_path = os.path.join(self.root, 'image')
        return len(os.listdir(imgs_path))

    def __getitem__(self, idx):
        # Here we have to return the item requested by `idx`. The PyTorch DataLoader class will use this method to make an iterable for training/validation loop.
        img_path = os.path.join(self.root, 'image', f'{str(idx)}.png')
        label_path = os.path.join(self.root, 'label', f'{str(idx)}.txt')

        # Import image
        image = np.transpose(torch.tensor(plt.imread(img_path)), (2, 0, 1))

        # Get label of corresponding image
        l = open(label_path, 'r')
        label = int(l.read())

        return image, label
  
DATA_train_path = LungDataset('./Dataset/Train', None)
DATA_test_path = LungDataset('./Dataset/Test', None)

mean = torch.tensor(0)
for i in range(0, len(DATA_train_path)):
  mean = mean + torch.mean(DATA_train_path[i][0])
  # print(mean)

print(f'Mean: {mean / len(DATA_train_path)}') 

std = torch.tensor(0)
for i in range(0, len(DATA_train_path)):
  std = std + torch.std(DATA_train_path[i][0])

print(f'Std: {std / len(DATA_train_path)}') 
