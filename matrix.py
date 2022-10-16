# coding=utf-8
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def confusion(true,pre):
    # sns.set()
    # f, ax = plt.subplots()
    y_true = np.loadtxt(true, delimiter=',')
    y_pred = np.loadtxt(pre, delimiter=',')  #loadtxt
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    print('true:',len(y_true))
    print('pred:',len(y_pred))

    C2 = confusion_matrix(y_true, y_pred)
    print(C2)  
    # sns.heatmap(C2, annot=True, ax=ax)  # drawing Heat Map
    plt.imshow(C2,  cmap=plt.cm.Oranges)
    indices = range(len(C2))
    #####################################################################
    plt.xticks(indices, ['H', 'IF7', 'IF14', 'IF21', 'BF7', 'BF14', 'BF21', 'unknown'])  # C0
    plt.yticks(indices, ['H', 'IF7', 'IF14', 'IF21', 'BF7', 'BF14', 'BF21', 'unknown'])

    plt.colorbar()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('confusion matrix')
    # display data
    for first_index in range(len(C2)):  # row
        for second_index in range(len(C2[first_index])):  # columns
            plt.text(first_index, second_index,  C2[second_index][first_index])
  
  
    plt.show()
    plt.savefig('ConfusionMatrix1.jpg')
    plt.close()
    ##########################################known and unknown 
    known = []
    for p in range(len(C2)):
        if p != len(C2) - 1:
            known.append(C2[p][p] / sum(C2[p]))
        else:
            unknown = C2[p][p] / sum(C2[p])
    known = sum(known) / (len(C2) - 1)
    print('known accary', known)
    print('unknown accary', unknown)

# if __name__ == '__main__':
#     confusion()
