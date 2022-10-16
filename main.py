#coding=utf-8
"""Main file to test the OSDN."""
import scipy
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.python.client import device_lib
# from utilss.nepali_characters import split
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from utils.openmax import get_train_test, create_model,get_pre_data
from utils.openmax import compute_activation, compute_openmax
import matrix
import train as ts
import matplotlib.pyplot as plt
# np.seterr(divide='ignore',invalid='ignore')
plt.switch_backend('agg')




def main():
    #####################################################################################################################
    # Step 1: Train a CNN model for the dataset you choice
    # Step 2: Load the trained model
    get_model = ts.train_model()
    model = load_model(get_model)
    #########################################################################################################################
    # Step 3: Load the training data you trained the CNN model
    data = get_train_test()
    x_train, x_test, y_train, y_test = data
    category = len(y_test[0])
    #########################################################################################################################
    # Step 4: Create a mean activation vector (MAV) and do weibull fit model
    create_model(model, data)
    ##########################################################################################################################
    # Step 6: Test the trained model to datasets from different distribution
    #test dataset
    x_pre,y_label = get_pre_data()
    data = x_pre,y_label
    actual_list = []
    ##################################################
    for i in y_label:
        if i < category:
            actual_list.append(i)
        elif i >= category:
            actual_list.append(10)
    ###########################################Compute openmax activation#####################################
    activation = compute_activation(model, x_pre)
    fc = activation['fc8']
    scipy.io.savemat('p230fc.mat', mdict={'p230fc': fc})
    softmax, openmax = compute_openmax(model, activation)
    ########################################################################################################################
    openmax_list = []
    for p in openmax:
        openmax_label = np.argmax(p)
        if openmax_label != category:
            openmax_list.append(openmax_label)
        else:
            openmax_list.append(10)
#############################################################################################################
    softmax_list = []
    for q in softmax:
        softmax_label = np.argmax(q)
        softmax_list.append(softmax_label)
###################################################Confusion matrix ( saving actual and predicted labels )####################################################################
    file = open('actual.txt', 'w')
    for i in range(len(actual_list)):
        s = str(actual_list[i]).replace('[', '').replace(']', '') + ','  
        if i == len(actual_list) - 1:
            s = s.rstrip(',')
        file.write(s)
    file.close()
    print("The file was saved successfully.")

    file = open('openmax.txt', 'w')
    for i in range(len(openmax_list)):
        s = str(openmax_list[i]).replace('[', '').replace(']', '') + ','
        if i == len(openmax_list) - 1:
            s = s.rstrip(',')
        file.write(s)
    file.close()
    print("The file was saved successfully.")
    matrix.confusion('actual.txt', 'openmax.txt')
###################################################################################################################
    file = open('softmax.txt', 'w')
    for i in range(len(softmax_list)):
        s = str(softmax_list[i]).replace('[', '').replace(']', '') + ','  
        if i == len(softmax_list) - 1:
            s = s.rstrip(',')
        file.write(s)
    file.close()
    print("The file was saved successfully.")
    matrix.confusion('actual.txt', 'softmax.txt')

    precision, recall, fscore, _ = precision_recall_fscore_support(actual_list, openmax_list)
    print('precision:', precision)
    print('recall', recall)
    print('macro fscore: ', np.mean(fscore))


if __name__ == "__main__":
    for e in range(10):
        main()
