#--------------------------------------------------------------------------------------------------------------------------------------------------------
#IMPORT LIBRARIES
import torchvision
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from matplotlib import pyplot
import shutil
import random
import math
import argparse

# Process location of dataset
parser = argparse.ArgumentParser(description='Process image dataset.')
parser.add_argument('dataset_dir', type=str, help='Directory containing your dataset data')
args = parser.parse_args()
#--------------------------------------------------------------------------------------------------------------------------------------------------------
#PREPARE TRAINING AND TESTING DATA 

# location of folder containing all the original images, to be copied
copy_dir = args.dataset_dir

# location of folder containing all organized images, to be copied to
root_dir = './Dataset'

# all labels
# Each label has the total number of images to correctly calculate ratio of images
classesdir = [['Covid', 2168], ['Healthy', 758], ['Others', 1247]]

# Create image/label folders for train and test folders
train_images = os.path.join(root_dir, 'Train', 'image')
train_labels = os.path.join(root_dir, 'Train', 'label')

test_images = os.path.join(root_dir, 'Test', 'image')
test_labels = os.path.join(root_dir, 'Test', 'label')

os.makedirs(train_images)
os.makedirs(train_labels)

os.makedirs(test_images)
os.makedirs(test_labels)

train_ratio = 0.8 #80% of images are for training
test_ratio = 0.2 #20% of images are for testing

# Keep track of the number of images for a certain class
img_count = 0

train_count = 0
test_count = 0

for cls in classesdir:
    # Loop through patient folders in class
    # Path of class folder
    class_path = os.path.join(copy_dir, cls[0])

    img_count = 0 # Reset image count for each class (to be used for train/test ratio)
    for patient in os.listdir(class_path):
        # Path of patient's folder
        patient_path = os.path.join(copy_dir, cls[0], patient)
        # Loop through images in patient folder
        for img in os.listdir(patient_path):
            # Path of patient's image
            img_path = os.path.join(patient_path, img)

            # Check if image count is less than 80%. If so, put in train
            if (img_count <= math.floor(cls[1] * train_ratio)):
                # Copy file from dataset to coerced dataset folder
                image_path = os.path.join(train_images, f'{int(train_count)}.png')
                shutil.copyfile(img_path, image_path)

                # Create label file corresponding to image
                label_path = os.path.join(train_labels, f'{int(train_count)}.txt')
                label = open(label_path, 'a')
                label.write(cls[0])
                label.close()
#Check stuff below --------------------------------------------------------------------------------------------------------------------
                # Rotate image 90, 180, 270 degrees
                f'{int(train_count)}90.png' = transforms.functional.rotate(f'{int(train_count)}.png',90)
                f'{int(train_count)}180.png' = transforms.functional.rotate(f'{int(train_count)}.png',180)
                f'{int(train_count)}270.png' = transforms.functional.rotate(f'{int(train_count)}.png',270)

                # Create label file corresponding to image (90 degree rotate)
                label_path = os.path.join(train_labels, f'{int(train_count)}90.txt')
                label = open(label_path, 'a')
                label.write(cls[0])
                label.close()

                # Create label file corresponding to image (180 degree rotate)
                label_path = os.path.join(train_labels, f'{int(train_count)}180.txt')
                label = open(label_path, 'a')
                label.write(cls[0])
                label.close()

                # Create label file corresponding to image (270 degree rotate)
                label_path = os.path.join(train_labels, f'{int(train_count)}270.txt')
                label = open(label_path, 'a')
                label.write(cls[0])
                label.close()
#Check stuff above --------------------------------------------------------------------------------------------------------------------

                # For file name
                train_count = train_count + 1
            else: # Otherwise, put the rest of the images in test
                # Copy file from dataset to coerced dataset folder
                image_path = os.path.join(test_images, f'{int(test_count)}.png')
                shutil.copyfile(img_path, image_path)

                # Create label file corresponding to image
                label_path = os.path.join(test_labels, f'{int(test_count)}.txt')
                label = open(label_path, 'a')
                label.write(cls[0])
                label.close()

                # For file name
                test_count = test_count + 1
            
            img_count = img_count + 1 # For ratio of train/test

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# allFileNames = os.listdir(src)

# #np.random.shuffle(allFileNames)
# train_FileNames,test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*(train_ratio)),int(len(allFileNames)*(test_ratio))])
#     #numpy.split(ary, indices_or_sections, axis) --> ary = input to be split, indices or sections- integer indicating size, axis, defaults to 0)

# train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
# test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

# Loop through class dirs
# Loop through each patient 

# print('Total images: ', len(allFileNames))
# print('Training: ', len(train_FileNames))
# print('Testing: ', len(test_FileNames))
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


