import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
import os
import numpy as np
import matplotlib.pyplot as plt
import math

# TODO: Construct your data in the following baseline structure: 1) ./Dataset/Train/image/, 2) ./Dataset/Train/label, 3) ./Dataset/Test/image, and 4) ./Dataset/Test/label
class LungDataset(Dataset):
    def __init__(self, root):        
        self.root = root

    def __len__(self):
        # Return number of points in the dataset based on root path
        imgs_path = os.path.join(self.root, 'image')
        return len(os.listdir(imgs_path))

    def __getitem__(self, idx):
        # Here we have to return the item requested by `idx`. The PyTorch DataLoader class will use this method to make an iterable for training/validation loop.
        # File names are based on idx.
        img_path = os.path.join(self.root, 'image', f'{str(idx)}.png')
        label_path = os.path.join(self.root, 'label', f'{str(idx)}.txt')

        # Import image
        # Transpose to be 3x244x244
        image = np.transpose(torch.tensor(plt.imread(img_path)), (2, 0, 1))

        # Normalize image to reduce computation
        # image = transforms.Normalize(parameters['mean'], parameters['std']).forward(image)\
        
        # Greyscale image
        # image = transforms.Grayscale(num_output_channels=3).forward(image)
        
        # Get label of corresponding image
        l = open(label_path, 'r')
        label = int(l.read())

        # Return manipulated image and label
        return image, label
  
fold = [
  LungDataset('./Dataset/fold0'),
  LungDataset('./Dataset/fold1'),
  LungDataset('./Dataset/fold2'),
  LungDataset('./Dataset/fold3'),
  LungDataset('./Dataset/fold4')
]

mean = torch.tensor(0)


for i in range(5):
  # Copy fold array
  training = fold.copy()

  testing = DataLoader(training[i], batch_size=32, shuffle=True)

  del training[i]

  training_count = 0

  for dataset in range(0, len(training)):
    training_count = training_count + len(training[dataset])

  print(training_count)

  mean = torch.tensor(0)
  std = torch.tensor(0)
  for dataset in range(0, len(training)):
    for i in range(0, len(training[dataset])):
      mean = mean + torch.mean(training[dataset][i][0])
      std = std + torch.std(training[dataset][i][0])
      # print(mean)

  print(f'{i} Mean: {mean / training_count}') 
  print(f'{i} Std: {std / training_count}') 
