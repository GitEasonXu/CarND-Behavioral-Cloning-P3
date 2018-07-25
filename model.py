
# coding: utf-8

# In[1]:
import re

import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn
import random



# select data source(s) here
# select data source(s) here
using_my_data = True
using_my_data_2 = True
using_my_data_3 = True
using_my_data_4 = True
using_my_data_5 = True
using_udacity_data = True
data_to_use = [using_my_data, using_my_data_2, using_my_data_3,using_my_data_4,using_my_data_5,using_udacity_data]
csv_path = ['./data/my_data_1/driving_log.csv', './data/my_data_2/driving_log.csv', './data/my_data_3/driving_log.csv', './data/my_data_4/driving_log.csv',  './data/my_data_5/driving_log.csv','./data/udacity_data/driving_log.csv']

# In[2]:

lines = []
for j in range(len(csv_path)):
    if data_to_use[j]:
        with open(csv_path[j]) as csv_file:
            data = csv.reader(csv_file)
            for line in data:
                lines.append(line)

                
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print(len(train_samples))


# In[3]:

def generator(samples, batch_size=16):
    num_samples = len(samples)
    while True:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                #center
                image_path = './data/IMG/'+ re.split(r"[\\/]",batch_sample[0])[-1]
                print(image_path)
                image = cv2.imread(image_path)
                images.append(image)
                image_flipped = cv2.flip(image,1)  #1 水平翻转 0 垂直翻转 -1 水平垂直翻转
                images.append(image_flipped)
                
                measurement = float(batch_sample[3])
                angles.append(measurement)
                measurement_flipped = -measurement
                angles.append(measurement_flipped)
                #left
                image_path = './data/IMG/'+ re.split(r"[\\/]",batch_sample[1])[-1]
                print(image_path)
                image = cv2.imread(image_path)
                images.append(image)
                image_flipped = cv2.flip(image,1)  #1 水平翻转 0 垂直翻转 -1 水平垂直翻转
                images.append(image_flipped)
                
                measurement = float(batch_sample[3]) + 0.2
                angles.append(measurement)
                measurement_flipped = -measurement
                angles.append(measurement_flipped)
                #right
                image_path = './data/IMG/'+ re.split(r"[\\/]",batch_sample[2])[-1]
                print(image_path)
                image = cv2.imread(image_path)
                images.append(image)
                image_flipped = cv2.flip(image,1)  #1 水平翻转 0 垂直翻转 -1 水平垂直翻转
                images.append(image_flipped)
                
                measurement = float(batch_sample[3]) - 0.2
                angles.append(measurement)
                measurement_flipped = -measurement
                angles.append(measurement_flipped)
            
            batch_x = np.array(images)
            batch_y = np.array(angles)
            
            yield sklearn.utils.shuffle(batch_x, batch_y)


# In[4]:

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)


# In[ ]:

import keras
from keras.layers import Flatten, Conv2D, Dense, Activation, Lambda, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras import optimizers
from keras import backend as K
from keras.layers.convolutional import Cropping2D

def train_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)),data_format='channels_last', input_shape=(160,320,3))) #output (90, 320, 3)
    model.add(Lambda(lambda x : x / 225. - 0.5))

    
    try:
        model.add(Lambda(lambda image:K.tf.image.resize_images(image, (90, 200))))
    except :
        #if you have older version of tensorflow
        model.add(Lambda(lambda image: K.tf.image.resize_images(image, 90, 200)))
    #model.add(Lambda(lambda x : x / 225. - 0.5))
    
    model.add(Conv2D(24,(5,5),strides=(2, 2), padding='valid',activation='relu'))
    model.add(Conv2D(36,(5,5),strides=(2, 2), padding='valid',activation='relu'))
    model.add(Conv2D(48,(5,5),strides=(2, 2), padding='valid',activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3),strides=(2, 2), padding='valid',activation='relu'))
    model.add(Conv2D(64,(3,3),strides=(2, 2), padding='valid',activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # Add a fully connected output layer
    model.add(Dense(1))
    
    
    adam = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='mse', optimizer=adam)
    #model.compile(loss='mse', optimizer=sgd)
    return model


def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    try:
        model.add(Lambda(lambda image:K.tf.image.resize_images(image, (66, 200))))
    except :
        # if you have older version of tensorflow
        model.add(Lambda(lambda image: K.tf.image.resize_images(image, 66, 200)))
    model.add(Conv2D(24,(5,5),strides=(2, 2), padding='valid',activation='relu'))
    model.add(Conv2D(36,(5,5),strides=(2, 2), padding='valid',activation='relu'))                                   
    model.add(Conv2D(48,(5,5),strides=(2, 2), padding='valid',activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64,(3,3),strides=(1, 1), padding='valid',activation='relu'))
    model.add(Conv2D(64,(3,3),strides=(1, 1), padding='valid',activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    #adam = optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer='adam')
    return model


model = nVidiaModel()
model.summary()

#model = train_model2()


# In[ ]:
print('\nTraining ------------')    #从文件中提取参数，训练后存在新的文件中
cm = 0                              #修改这个参数可以多次训练  
if cm >= 1:
    model.load_weights('model_{}.h5'.format(cm))
    
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples) // 128, epochs=10, verbose=1,
                 validation_data=validation_generator, validation_steps=len(validation_samples) // 128)

    #model.fit(features, labels, batch_size=batch_size , epochs=epochs, validation_split=0.3)
    #model.fit(features, labels,batch_size=128, epochs=3, validation_split=0.3)
cm += 1
model.save('model_{}.h5'.format(cm))
print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("loss.png")

#plt.show()
#cm += 1
#model.save('model_{}.h5'.format(cm))
print("Model Saved!")


# In[ ]:




# In[ ]:



