#coding=utf-8
import scipy.io
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K


from utils.evt_fitting import weibull_tailfitting
from utils.compute_openmax import recalibrate_scores

import scipy.spatial.distance as spd

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


label = [0,1,2,3,4,5,6]#CWRU
def seperate_data(x, y):
    ind = y.argsort()
    sort_x = x[ind[::-1]]
    sort_y = y[ind[::-1]]
    dataset_x = []
    dataset_y = []
    mark = 0
    for a in range(len(sort_y)-1):
        if sort_y[a] != sort_y[a+1]:
            dataset_x.append(np.array(sort_x[mark:a+1]))
            dataset_y.append(np.array(sort_y[mark:a+1]))
            mark = a + 1
        if a == len(sort_y)-2:
            dataset_x.append(np.array(sort_x[mark:len(sort_y)+1]))
            dataset_y.append(np.array(sort_y[mark:len(sort_y)+1]))
    return dataset_x, dataset_y
def compute_feature(x, model):
    score = get_activations(model, 21, x)
    fc15 = get_activations(model, 20, x)
    return score, fc15


def compute_mean_vector(feature):
    return np.mean(feature, axis=0)


def compute_distances(mean_feature, feature, category_name):
    eucos_dist, eu_dist, cos_dist = [], [], []
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in feature:
        eu_dist += [spd.euclidean(mean_feature, feat)]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(
            mean_feature, feat)]
    # distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    return distances


def get_pre_data():
################################################################################################################
    xy_ = loadmat('dataset/CRWU/CRWU0/C0_0_.mat')  # crwu单工况
    xy = xy_['C0_0_']
    #######################################
    x_data = xy[:, :-1]
    y_label = (xy[:, [-1]]).reshape(-1, )  # array数组(n,1)转换成(n,)
    pre_x = np.expand_dims(x_data, axis=-1)
    return pre_x,y_label




def get_train_test():
    ######C0
    xy_ = loadmat('dataset/CRWU/CRWU0/C0_0.mat')  # crwu单工况
    xy = xy_['C0_0']
    #######################################
    x_data = xy[:, :-1]
    y_label = (xy[:,[-1]])
    y_label = (xy[:, [-1]]).reshape(-1, )
    category = list(set(y_label))
    def splitTrainTest(X, y, ratio=0.7):
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
    cate_seen_train_y = keras.utils.to_categorical(train_y, len(category))
    cate_seen_test_y = keras.utils.to_categorical(test_y, len(category))#p3-5

    return seen_train_X, seen_test_X_, cate_seen_train_y, cate_seen_test_y


def get_activations(model, layer, X_batch):
    # print (model.layers[6].output)
    a=[model.layers[0].input, K.learning_phase()]
    b=[model.layers[layer].output]
    get_activations = K.function(a,b)
    activations = get_activations([X_batch, 0])[0]
    # print (activations.shape)
    return activations


def get_correct_classified(pred, y):
    a = pred>0.5
    pred = a * 1
    res = np.all(pred == y, axis=1)
    return res


def create_model(model, data):

    x_train, x_test, y_train, y_test = data
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    pred = model.predict(x_all)
    index = get_correct_classified(pred, y_all)
    x1_test = x_all[index]
    y1_test = y_all[index]
    y1_test1 = y1_test.argmax(1)
    sep_x, sep_y = seperate_data(x1_test, y1_test1)


    feature = {}
    feature["score"] = []
    feature["fc15"] = []
    weibull_model = {}
    feature_mean1 = []
    feature_distance1 = []
    feature_mean = []
    feature_distance = []

    known = []
    for i in range(len(sep_y)):
        print(i, sep_x[i].shape)
        weibull_model[label[i]] = {}
        #################################################################################################################
        score, fc15 = compute_feature(sep_x[i], model)
        for i in fc15:
            a = np.argmax(i)
            known.append(i[a])
        mean = compute_mean_vector(fc15)

        distance = compute_distances(mean, fc15, sep_y)
        feature_mean1.append(mean)
        feature_distance1.append(distance)
    known = np.array(known)
    scipy.io.savemat('known.mat', mdict={'Known': known})
    for i in range(0, len(feature_mean1)):
        feature_mean.append(feature_mean1[len(feature_mean1) - i - 1])
    for j in range(0, len(feature_distance1)):
        feature_distance.append(feature_distance1[len(feature_distance1) - j - 1])
    np.save('mean', feature_mean)
    np.save('distance', feature_distance)


def build_weibull(mean, distance, tail):
    weibull_model = {}
    for i in range(len(mean)):
        weibull_model[label[i]] = {}
        weibull = weibull_tailfitting(mean[i], distance[i], tailsize=tail)
        weibull_model[label[i]] = weibull
    return weibull_model


def compute_openmax(model, imagearr):
    mean = np.load('mean.npy', allow_pickle=True)
    distance = np.load('distance.npy', allow_pickle=True)
    # Use loop to find the good parameters
    alpharank_list = [1]  # C3/C2
    tail_list = [2]

    for alpha in alpharank_list:
        weibull_model = {}
        openmax = None
        softmax = None
        for tail in tail_list:
            weibull_model = build_weibull(mean, distance, tail)
            openmax, softmax= recalibrate_scores(
                weibull_model, label, imagearr, alpharank=alpha)
    return softmax, openmax


def process_input(model, ind, data):
    x_train, x_test, y_train, y_test = data
    imagearr = {}
    plt.imshow(np.squeeze(x_train[ind]))
    plt.show()
    image = np.reshape(x_train[ind], (1, 28, 28, 1))
    score5, fc85 = compute_feature(image, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    return imagearr


def compute_activation(model, img):
    imagearr = {}
    score5, fc85 = compute_feature(img, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    return imagearr


