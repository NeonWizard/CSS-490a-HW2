import keras, os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

# -- load images
id_generator = ImageDataGenerator()
# TODO: consider using color_mode parameter to reduce to grayscale
train_data = id_generator.flow_from_directory(
  directory="data/train", target_size=(224, 224))
test_data = id_generator.flow_from_directory(
  directory="data/test", target_size=(224, 224))

# -- build model
model = Sequential()

# 2x convolution layer of 64 channel of 3x3 kernal and same padding
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# 2x convolution layer of 128 channel of 3x3 kernal and same padding
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# 3x convolution layer of 256 channel of 3x3 kernal and same padding
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# 3x convolution layer of 512 channel of 3x3 kernal and same padding
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# 3x convolution layer of 512 channel of 3x3 kernal and same padding
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=3, activation="softmax"))

# -- Compile model
opt = adam_v2.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

# --
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit(
  train_data,
  # steps_per_epoch=100,
  validation_data=test_data,
  # validation_steps=10,
  epochs=100,
  batch_size=8,
  callbacks=[checkpoint,early])