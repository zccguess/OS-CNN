# coding=utf-8
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
import numpy as np
from scipy.io import loadmat
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.keras.utils.vis_utils import plot_model
# import tensorflow
# from model import LeNet1d
from model import LeNetbbd
# import resnet
# from tensorflow_core.python.keras.layers import GlobalAveragePooling1D

def train_model():
    ########################################################phm任务#################################################################
    xy_ = loadmat('dataset/CRWU/CRWU0/C0_0.mat')
    xy = xy_['C0_0']
    ########################################################################################################################################
    x_data = xy[:, :-1]
    y_label = (xy[:, [-1]]).reshape(-1, )
    category = list(set(y_label))
    def splitTrainTest(X, y):
        shuffle_idx = np.random.permutation(len(y_label))
        shuffled_X = X[shuffle_idx]
        shuffled_y = y[shuffle_idx]
        split_idx = int(0.7 * len(y))
        return shuffled_X[:split_idx], shuffled_y[:split_idx], shuffled_X[split_idx:], shuffled_y[split_idx:]

    train_X, train_y, test_X, test_y = splitTrainTest(x_data, y_label)
    print('train.shape', train_X.shape, train_y.shape)
    print('test.shape', test_X.shape, test_y.shape)

    # numpy转为tensor
    seen_train_X = np.expand_dims(train_X, axis=-1)
    seen_test_X_ = np.expand_dims(test_X, axis=-1)
    #############
    cate_seen_train_y = to_categorical(train_y, len(category))
    cate_seen_test_y = to_categorical(test_y, len(category))
    filepath = 'b.h5'
    checkpointer = ModelCheckpoint(filepath)
    # model = LeNet1d.model1
    model = LeNetbbd.model1
    # model = resnet.model
    # train
    print("-----------------train-------------------")
    history = model.fit(seen_train_X, cate_seen_train_y, epochs=100, batch_size=32, callbacks=[checkpointer])##batch_size需要调整
    print("------------------test--------------------")
    loss, accuracy = model.evaluate(seen_test_X_, cate_seen_test_y)
    print('\ntest loss:', loss)
    print('\ntest accuracy:', accuracy)
    return filepath