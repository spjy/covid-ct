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
import cv2

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
                #Rotate images
                for i in range (90,270,90):
                    # Rotate image 90 degrees
                    first = cv2.rotate(image_path,i)
                    cv2.imwrite(image_path,first)
                    img_count = img_count + 1;

                    # Create label file corresponding to image (90 degree rotate)
                    label_path = os.path.join(train_labels, f'{int(train_count)}.txt')
                    label = open(label_path, 'a')
                    label.write(cls[0])
                    label.close()
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
                #Rotate Images
                for i in range (90,270,90):
                    # Rotate image 90 degrees
                    first = cv2.rotate(image_path,i)
                    cv2.imwrite(image_path,first)
                    img_count = img_count + 1;

                    # Create label file corresponding to image (90 degree rotate)
                    label_path = os.path.join(test_labels, f'{int(test_count)}.txt')
                    label = open(label_path, 'a')
                    label.write(cls[0])
                    label.close()
                # For file name
                test_count = test_count + 1
            
            img_count = img_count + 1 # For ratio of train/test

#--------------------------------------------------------------------------------------------------------------------------------------------------------


