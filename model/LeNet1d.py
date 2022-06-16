# # coding=utf-8
# #############################################################################################################################################
# plot_model(model1, to_file='model2.png', show_shapes=True)
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Activation, MaxPool1D, Flatten, BatchNormalization, Dropout
from tensorflow import keras

def LeNet(filters, kernel_size, strides, input_layer,padding):
    x = Conv1D(filters, kernel_size, strides=strides, padding=padding)(input_layer)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=2, strides=1,padding='valid')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    return x

inputs = Input(shape=(2049, 1), name="inputs")
# inputs = Input(shape=(4097, 1), name="inputs")  # phm
conv1 = LeNet(16, 64, 16, inputs,padding='same')
conv2 = LeNet(32, 3,2, conv1,padding='same')
conv3 = LeNet(64, 3, 2, conv2,padding='same')
conv4 = LeNet(64, 3, 2, conv3,padding='same')
conv5= LeNet(64, 3, 2, conv4,padding='valid')
# x = GlobalAveragePooling1D()(conv4)#S0S1
# x = Dropout(0.3)(x)#S0S1
# conv6= LeNet(64, 3, 1, conv5,padding='valid')
x = Flatten()(conv5)
x = Dense(7)(x)
output = Activation('sigmoid')(x)
# output = Activation('softmax')(x)

model1 = keras.models.Model(inputs=inputs, outputs=output)
model1.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
