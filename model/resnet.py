# coding=utf-8
# from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout, Input, ZeroPadding1D, AveragePooling1D, Softmax, \
    BatchNormalization, Activation, Add, GlobalAveragePooling1D


def Conv_BN_Relu(nb_filter, kernel_size, strides, input_layer):
    x = Conv1D(nb_filter, kernel_size, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
# ResNet18 Network corresponding residual module a and residual module b
def resiidual_a_or_b(input_x,nb_filter, flag):
    if flag == 'a':
        # main road
        x = Conv_BN_Relu(nb_filter, 3, 1, input_x)
        x = Conv_BN_Relu(nb_filter,3, 1, x)
        # output
        y = Add()([x, input_x])
        return y
    elif flag == 'b':
        # main road
        x = Conv_BN_Relu(nb_filter, 3, 2, input_x)
        x = Conv_BN_Relu(nb_filter, 3, 1, x)
        # The branch down sampling
        input_x = Conv_BN_Relu(nb_filter, 3, 2, input_x)
        # output
        y = Add()([x, input_x])
        return y
# model = Sequential()
# Last layer
input_layer = Input(shape=(2049,1))
print(input_layer)
# x = ZeroPadding1D(1)(input_layer)
# conv1_x

conv1 = Conv_BN_Relu(16, 64,16,input_layer)
conv1_Maxpooling = MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)
# conv2_x
x = resiidual_a_or_b(conv1_Maxpooling,32,'b')
x = resiidual_a_or_b(x,  32,'a')
# conv3_x
x = resiidual_a_or_b(x, 64, 'b')
x = resiidual_a_or_b(x,  64, 'a')
# conv4_x
x = resiidual_a_or_b(x,  64, 'b')
x = resiidual_a_or_b(x, 64, 'a')
x = resiidual_a_or_b(x,  64, 'b')
x = resiidual_a_or_b(x, 64, 'a')
x = resiidual_a_or_b(x,  64, 'b')
x = resiidual_a_or_b(x, 64, 'a')
# Last layer
x = GlobalAveragePooling1D()(x)
# x = AveragePooling1D(pool_size=1)(x)
x = Flatten()(x)
x = Dense(64*3)(x)
x = Dense(4)(x)
x = Dropout(0.5)(x)
x = Activation('sigmoid')(x)
model = Model(inputs=input_layer, outputs=x)
# plot_model(model, to_file='one_rest/network/model_classifier.png', show_shapes=True)
print(model.summary())
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
