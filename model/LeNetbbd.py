# coding=utf-8
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Activation, MaxPool1D, Flatten, BatchNormalization, Dropout
from tensorflow import keras
import tensorflow as tf
# -----------------------input size>=32---------------------------------
def LeNet(filters, kernel_size, strides, input_layer):
    x = Conv1D(filters, kernel_size, strides=strides,padding='valid',activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=1, strides=1)(x)
    return x
inputs = Input(shape=(2049, 1), name="inputs")
# inputs = Input(shape=(4097, 1), name="inputs")#phm
conv1 = LeNet(16, 64, 16, inputs)
conv2 = LeNet(32, 3,2, conv1)
conv3 = LeNet(64, 3, 2, conv2)
conv4 = LeNet(64, 3, 2, conv3)
conv5 = LeNet(64, 3, 2, conv4)
conv6 = LeNet(64, 3, 2, conv5)
x = Flatten()(conv6)
# x = BatchNormalization()(x)
x = Dense(7)(x)#CWRU
output = Activation('sigmoid')(x)
model1 = keras.models.Model(inputs=inputs, outputs=output)
a=[model1.layers[0].input]
b=['a',model1.layers[14].output]
c=['b',model1.layers[15].output]
model1.summary()
model1.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
