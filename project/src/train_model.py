import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras import optimizers

import cv2

data_path = "dataset_train"
model_file = "./model/captcha_model1.hdf5"
imagesize = 20
batchsize = 32

data_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


train_generator = data_datagen.flow_from_directory(
	data_path,
	target_size=(imagesize,imagesize),
	batch_size=batchsize,
	color_mode='grayscale',
	subset='training'
	)
valid_generator = data_datagen.flow_from_directory(
	data_path,
	target_size=(imagesize,imagesize),
	batch_size=batchsize,
	color_mode='grayscale',
	subset='validation'
	)

model = Sequential()
model.add(Conv2D(20,(5,5),padding='same',input_shape=(20,20,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Conv2D(50,(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(33,activation='softmax'))

model.compile(loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

model.fit_generator(train_generator,steps_per_epoch=32930/batchsize,
	validation_data=valid_generator,validation_steps=8214/batchsize,
	epochs=10,verbose=1)

print(model.summary())
model.save(model_file)
