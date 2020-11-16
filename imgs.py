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
import imutils

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

def process_image(image_original, image_directory, label_directory, filename, rotation):
    if (image_original.endswith('.png')):
        # Copy file from dataset to coerced dataset folder
        image_path = os.path.join(image_directory, f'{int(filename)}.png')
        shutil.copyfile(image_original, image_path)

        img = cv2.imread(image_original)

        # Rotate image
        rot = imutils.rotate_bound(img, rotation)
        cv2.imwrite(image_path, rot)

        # Create label file corresponding to image (90 degree rotate)
        label_path = os.path.join(label_directory, f'{int(filename)}.txt')
        label = open(label_path, 'a')
        label.write(cls[0])
        label.close()

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
                #Rotate images
                for rotation in range (0, 271, 90):
                    process_image(img_path, train_images, train_labels, train_count, rotation)
                    
                    # For file name
                    train_count = train_count + 1
            else: # Otherwise, put the rest of the images in test
                # Rotate Images
                for rotation in range (0, 271, 90):
                    process_image(img_path, test_images, test_labels, test_count, rotation)

                    test_count = test_count + 1
            
            print(img_count)
            img_count = img_count + 1 # For ratio of train/test

#--------------------------------------------------------------------------------------------------------------------------------------------------------


