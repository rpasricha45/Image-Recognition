# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:51:43 2018

@author: rpasr
"""
# import libraires
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Constants
# AMOUNT of data 
AMOUNTOFPICS = 10000
TESTSET = 2000
TRAININGSET = AMOUNTOFPICS-2000

# initalise CNN

classifier = Sequential()

# Convolution layer 

classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3 ), activation = "relu"))



# max pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

#TODO: IF NEED MORE ACCURACY ADD ANOTHER LAYER
# faltning
classifier.add(Flatten())

# full conecction like ANN
classifier.add(Dense(output_dim = 128,activation = "relu"))
classifier.add(Dense(output_dim = 1,activation = "sigmoid"))
# complile CNN

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics =['accuracy'])

# Proces and fit the images to the CNN

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=TRAININGSET,
        epochs=2,
        validation_data=test_set,
        validation_steps=TESTSET)