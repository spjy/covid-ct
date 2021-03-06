import os
import shutil
import random
import math

# Randomize train/test

# Create image/label folders for train and test folders

root_dir = os.path.join('Dataset')
sorted_dataset = os.path.join('Dataset')

train_ratio = 0.8 #80% of images are for training
test_ratio = 0.2 #20% of images are for testing

covid_images = os.path.join(sorted_dataset, 'Covid', 'image')
covid_labels = os.path.join(sorted_dataset, 'Covid', 'label')

healthy_images = os.path.join(sorted_dataset, 'Healthy', 'image')
healthy_labels = os.path.join(sorted_dataset, 'Healthy', 'label')

other_images = os.path.join(sorted_dataset, 'Others', 'image')
other_labels = os.path.join(sorted_dataset, 'Others', 'label')

# 5 random
for i in range(5):
  fold_images = os.path.join(root_dir, f'fold{i}', 'image')
  fold_labels = os.path.join(root_dir, f'fold{i}', 'label')

  try:
    os.makedirs(fold_images)
    os.makedirs(fold_labels)

  except:
    os.rmdir(fold_images)
    os.rmdir(fold_labels)

# train_images = os.path.join(root_dir, 'Train', 'image')
# train_labels = os.path.join(root_dir, 'Train', 'label')

# test_images = os.path.join(root_dir, 'Test', 'image')
# test_labels = os.path.join(root_dir, 'Test', 'label')

# try:
#   os.makedirs(train_images)
#   os.makedirs(train_labels)

#   os.makedirs(test_images)
#   os.makedirs(test_labels)
# except:
#   os.rmdir(train_images)
#   os.rmdir(train_labels)

#   os.rmdir(test_images)
#   os.rmdir(test_labels)

# Total number in each class
numbers = [8668, 3028, 4988]

#
class_entries = []
class_count = 0
train_img_count = 0
test_img_count = 0

classesdir = [['Covid', 8667, '0'], ['Healthy', 3027, '1'], ['Others', 4987, '2']]

fold_counts = [0, 0, 0, 0, 0]

for cls in classesdir:
  print(f'Processing {cls[0]} dataset')
  # Loop through patient folders in class
  # Path of class folder
  class_img_count = 0 # Reset image count for each class (to be used for train/test ratio)
  class_entries = []
  class_count = 0

  # Array with random indices for class
  five_fold = []

  # Generates random order of numbers
  while (len(five_fold) != cls[1]):
    k = random.randint(0, cls[1])

    if (k not in five_fold):
      five_fold.append(k)

  # Keep track of which fold we are on
  fold_iteration = 0

  split_amount = math.floor(cls[1] / 5)

  # Loop through array of random 
  for count, elem in enumerate(five_fold):
    # Get original file to copy
    image_directory = os.path.join(sorted_dataset, cls[0], 'image', f'{elem}.png')
    label_directory = os.path.join(sorted_dataset, cls[0], 'label', f'{elem}.txt')

    # New copied over file in fold folder
    fold_images = os.path.join(root_dir, f'fold{fold_iteration}', 'image', f'{fold_counts[fold_iteration]}.png')
    fold_labels = os.path.join(root_dir, f'fold{fold_iteration}', 'label', f'{fold_counts[fold_iteration]}.txt')

    # Copy files
    shutil.copyfile(image_directory, fold_images)
    shutil.copyfile(label_directory, fold_labels)

    # image name counter for fold
    fold_counts[fold_iteration] = fold_counts[fold_iteration] + 1

    # increment iteration
    if (count == split_amount + (split_amount * fold_iteration)):
      fold_iteration = fold_iteration + 1

    if (fold_iteration == 5):
      break
