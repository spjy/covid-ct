import os
import shutil
import random
import math

# Randomize train/test

# Create image/label folders for train and test folders

root_dir = os.path.join('Dataset')
sorted_dataset = os.path.join('home', 'spjy', 'SortedDataset')

train_ratio = 0.8 #80% of images are for training
test_ratio = 0.2 #20% of images are for testing

covid_images = os.path.join(sorted_dataset, 'Covid', 'image')
covid_labels = os.path.join(sorted_dataset, 'Covid', 'label')

healthy_images = os.path.join(sorted_dataset, 'Healthy', 'image')
healthy_labels = os.path.join(sorted_dataset, 'Healthy', 'label')

other_images = os.path.join(sorted_dataset, 'Others', 'image')
other_labels = os.path.join(sorted_dataset, 'Others', 'label')

# 5 random
for i in range(4):
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

numbers = [8668, 3028, 4988]

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

  five_fold = []

  while (len(five_fold) != cls[1]):
    k = random.randint(0, cls[1])

    if (k not in five_fold):
      five_fold.append(k)

  fold_iteration = 0

  for count, elem in enumerate(five_fold):
    image_directory = os.path.join(root_dir, cls[0], 'image', f'{elem}.png')
    label_directory = os.path.join(root_dir, cls[0], 'label', f'{elem}.txt')

    fold_images = os.path.join(root_dir, f'fold{fold_iteration}', 'image', f'{fold_counts[fold_iteration]}.png')
    fold_labels = os.path.join(root_dir, f'fold{fold_iteration}', 'label', f'{fold_counts[fold_iteration]}.png')

    shutil.copyfile(image_directory, fold_images)
    shutil.copyfile(label_directory, fold_labels)

    # image names
    fold_counts[fold_iteration] = fold_counts[fold_iteration] + 1

    if (count == math.floor(cls[1] / 5)):
      fold_iteration = fold_iteration + 1


  # print(class_count)

  # for img in os.listdir(os.path.join(root_dir, cls[0], 'image')): 
  #   k = int(img.split('.')[0])
  #   if k not in class_entries:
  #     image_directory = os.path.join(root_dir, cls[0], 'image', f'{k}.png')
  #     label_directory = os.path.join(root_dir, cls[0], 'label', f'{k}.txt')

  #     test_images_new = os.path.join(test_images, f'{test_img_count}.png')
  #     test_labels_new = os.path.join(test_labels, f'{test_img_count}.txt')

  #     shutil.copyfile(image_directory, test_images_new)
  #     shutil.copyfile(label_directory, test_labels_new)

  #     test_img_count = test_img_count + 1


  # image_directory = os.path.join(root_dir, cls[0], 'image', f'{k}.png')
  # label_directory = os.path.join(root_dir, cls[0], 'label', f'{k}.txt')

  # train_images_new = os.path.join(train_images, f'{train_img_count}.png')
  # train_labels_new = os.path.join(train_labels, f'{train_img_count}.txt')

  # shutil.copyfile(image_directory, train_images_new)
  # shutil.copyfile(label_directory, train_labels_new)

  # class_entries.append(k)
  # train_img_count = train_img_count + 1
  # class_count = class_count + 1