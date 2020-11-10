#msg= "hello"
#print(msg)
#to run - python test.py
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#IMPORT LIBRARIES
import torchvision
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from matplotlib import pyplot
#dataset = datasets.ImageFolder('path', transform=transform)
#print(os.listdir("../input")) 
import shutil
import random
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#PREPARE TRAINING AND TESTING DATA 

# Creating Train / Val / Test folders (One time use)
rootdir = 'Dataset copy' # location of folder containing all images 
classesdir = ['Covid','Healthy','Other'] #all labels 

train_ratio = 0.8 #80% of images are for training
test_ratio = 0.2 #20% of images are for testing
for cls in classesdir:
    os.makedirs(root_dir +'/train/' + cls) #Making test and training folders for each label?? 
    os.makedirs(root_dir +'/test/' + cls)

# Creating partitions of the data after shuffeling
src = root_dir + cls # Folder to copy images from

allFileNames = os.listdir(src)
#np.random.shuffle(allFileNames)
train_FileNames,test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*(train_ratio)),int(len(allFileNames)*(test_ratio)]))
    #numpy.split(ary, indices_or_sections, axis) --> ary = input to be split, indices or sections- integer indicating size, axis, defaults to 0)

train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Testing: ', len(test_FileNames))
#--------------------------------------------------------------------------------------------------------------------------------------------------------

#APPLY TRANSFROMATIONS

#Define transforms (0,90,180,270 degree rotations)
#transform1 = transforms.Compose(transforms.ToTensor()])
#transform2 = transforms.Compose([transforms.RandomRotation(90),transforms.ToTensor()])
#transform3 = transforms.Compose([transforms.RandomRotation(180),transforms.ToTensor()])
#transform4 = transforms.Compose([transforms.RandomRotation(270),transforms.ToTensor()])

#Apply 
#train_data = datasets.ImageFolder(root_dir + ‘/train + cls’,transform1=train_transforms, 
#transform2=train_transforms, transform3=train_transforms, transform4=train_transforms)    
                                  
#test_data = datasets.ImageFolder(root_dir + ‘/test’ + cls,transform1=test_transforms, 
#transform2=test_transforms, transform3=test_transforms, transform4=test_transforms)
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#Data Loading
#trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
#testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


