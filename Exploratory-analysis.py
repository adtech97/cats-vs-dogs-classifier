base_dir = '../Datasets/dogs-vs-cats/'

import os

training_data_path = os.path.join(base_dir, 'train')
test_path = os.path.join(base_dir, 'test1')

# Prepare paths for train and test data
print(training_data_path)
print(test_path)

# get the counts for each
print(len(os.listdir(training_data_path)))
print(len(os.listdir(test_path)))

import matplotlib.image as img

# print the shape of images first few images.
train_image = [img for img in os.listdir(training_data_path)]
print(train_image[1])
image = img.imread(os.path.join(training_data_path, train_image[3]))
print(image.shape)
