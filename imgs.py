#--------------------------------------------------------------------------------------------------------------------------------------------------------
#IMPORT LIBRARIES
import os
import shutil
import math
import argparse
import cv2
import imutils
import random
import numpy as np

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
classesdir = [['Covid', 2167, '0'], ['Healthy', 757, '1'], ['Others', 1246, '2']]

img_total = 16684

covid_images = os.path.join(root_dir, classesdir[0][0], 'image')
covid_labels = os.path.join(root_dir, classesdir[0][0], 'label')

healthy_images = os.path.join(root_dir, classesdir[1][0], 'image')
healthy_labels = os.path.join(root_dir, classesdir[1][0], 'label')

other_images = os.path.join(root_dir, classesdir[2][0], 'image')
other_labels = os.path.join(root_dir, classesdir[2][0], 'label')

os.makedirs(covid_images)
os.makedirs(covid_labels)

os.makedirs(healthy_images)
os.makedirs(healthy_labels)

os.makedirs(other_images)
os.makedirs(other_labels)

train_ratio = 0.8 #80% of images are for training
test_ratio = 0.2 #20% of images are for testing

train_amt = math.floor(img_total * train_ratio)
test_amt = math.floor(img_total * test_ratio)

# Keep track of the number of images for a certain class
img_count = 0
class_img_count = 0

train_count = 0
test_count = 0

for cls in classesdir:
    print(f'Processing {cls[0]} dataset')
    # Loop through patient folders in class
    # Path of class folder
    class_path = os.path.join(copy_dir, cls[0])

    class_img_count = 0 # Reset image count for each class (to be used for train/test ratio)

    for patient in os.listdir(class_path):
        # Path of patient's folder
        patient_path = os.path.join(copy_dir, cls[0], patient)
        # Loop through images in patient folder
        if os.path.isdir(patient_path):
            for img in os.listdir(patient_path):
                # Path of patient's image
                if (img.endswith('.png')):
                    img_path = os.path.join(patient_path, img)

                    #Rotate images
                    for rotation in range (0, 271, 90):
                        k = random.randint(0, 1) # 0 = train, 1 = test

                        image_directory = os.path.join(root_dir, cls[0], 'image')
                        label_directory = os.path.join(root_dir, cls[0], 'label')
                        filename = class_img_count

                        # Copy file from dataset to coerced dataset folder
                        image_path = os.path.join(image_directory, f'{int(filename)}.png')
                        shutil.copyfile(img_path, image_path)

                        img = cv2.imread(img_path)
                        # Resize image to fit 
                        rsz = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                        # Rotate image
                        rot = imutils.rotate_bound(rsz, rotation)

                        final_image = cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)
                        
                        # Grayscale
                        cv2.imwrite(image_path, final_image)

                        # Create label file corresponding to image (90 degree rotate)
                        label_path = os.path.join(label_directory, f'{int(filename)}.txt')
                        label = open(label_path, 'a')
                        label.write(cls[2])
                        label.close()

                        img_count = img_count + 1
                        class_img_count = class_img_count + 1

    print(class_img_count)

#--------------------------------------------------------------------------------------------------------------------------------------------------------

