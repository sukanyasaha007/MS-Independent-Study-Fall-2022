import os
import csv
from helper import plot
from cnn_architectures import nvidia_model

from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

image_datapath = "../data"

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # read in images and steering angles for the batch
            for batch_sample in batch_samples:
                center_image = cv2.imread((image_datapath+batch_sample[0]).replace(' ',''))
                # print(image_datapath+batch_sample[0].replace(' ',''))
                left_image = cv2.imread((image_datapath+batch_sample[1]).replace(' ',''))
                right_image = cv2.imread((image_datapath+batch_sample[2]).replace(' ',''))
                correction = 0.2
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction # correction for treating left images as center
                right_angle = center_angle - correction # correction for treating right images as center
                images.extend([center_image[70:135, :],left_image[70:135, :],right_image[70:135, :]]) # cropping images
                angles.extend([center_angle,left_angle,right_angle])

            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1)) # flipping image for data augmentation
                augmented_angles.append(angle * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

row, col, ch = 65, 320, 3

print('Importing keras libraries')

model = nvidia_model(row, col, ch)

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples*6), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
# note, the total training samples are 6 times per epoch counting both original
# and flipped left, right and center images
model.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plot(history_object)