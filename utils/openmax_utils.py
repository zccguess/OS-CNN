import scipy.spatial.distance as spd
import numpy as np


def parse_synsetfile(synsetfname):
    """ Read ImageNet 2012 file
    """
    categorylist = open(synsetfname, 'r').readlines()
    imageNetIDs = {}
    count = 0
    for categoryinfo in categorylist:
        wnetid = categoryinfo.split(' ')[0]
        categoryname = ' '.join(categoryinfo.split(' ')[1:])
        imageNetIDs[str(count)] = [wnetid, categoryname]
        count += 1

    assert len(imageNetIDs.keys()) == 1000
    return imageNetIDs


def getlabellist(synsetfname):
    """ read sysnset file as python list. Index corresponds to the output that
    caffe provides
    """

    categorylist = open(synsetfname, 'r').readlines()
    labellist = [category.split(' ')[0] for category in categorylist]
    return labellist


def compute_distance( query_channel, channel, mean_vec,
        distance_type='eucos'):
    """ Compute the specified distance type between chanels of mean vector
    and query image. In caffe library, FC8 layer consists of 10 channels.
    Here, we compute distance of distance of each channel (from query image)
    with respective channel of Mean Activation Vector. In the paper,
    we considered a hybrid distance eucos which
    combines euclidean and cosine distance for bouding open space.
    Alternatively,
    other distances such as euclidean or cosine can also be used.

    Input:
    --------
    query_channel: Particular FC8 channel of query image
    channel: channel number under consideration
    mean_vec: mean activation vector

    Output:
    --------
    query_distance : Distance between respective channels

    """
    # print ('copute',query_channel,channel,mean_vec)
    # print ('mean',mean_vec)
    # print ('query_ch',query_channel)
    # print ('channel',channel)

    query_channel = np.array(query_channel)###Particular FC8 channel of query data[600,5]
    # mean_vec = np.reshape(mean_vec, (7, 1))  # CWRU
    # mean_vec = np.reshape(mean_vec, (3, 1))#J0-J1
    # mean_vec = np.reshape(mean_vec, (2, 1))  # J2-J4
    mean_vec = np.reshape(mean_vec, (len(mean_vec), 1))#p1-2#[5,1]
    # mean_vec = np.reshape(mean_vec, (4, 1))  #p3-5
    # print ('shape',mean_vec.shape)
    # exit()
    # print (distance_type)
    # if distance_type == 'eucos':
    #     # print (mean_vec.shape,query_channel.shape)
    #     query_distance = spd.euclidean(
    #         mean_vec, query_channel)/200. + spd.cosine(mean_vec, query_channel)
    # elif distance_type == 'euclidean':
    #     query_distance = spd.euclidean(mean_vec, query_channel)/200.
    # elif distance_type == 'cosine':
    #     query_distance = spd.cosine(mean_vec, query_channel)
    # else:
    #     print(
    #         "distance type not known: enter either of eucos," +
    #         " euclidean or cosine")
    query_distance_list = []
    for i in query_channel:
        if distance_type == 'eucos':
            # print (mean_vec.shape,query_channel.shape)
            query_distance = spd.euclidean(
                mean_vec, i) / 200. + spd.cosine(mean_vec, i)
            query_distance_list.append(query_distance)
        elif distance_type == 'euclidean':
            query_distance_list.append(spd.euclidean(mean_vec, i) / 200.)
        elif distance_type == 'cosine':
            query_distance_list = spd.cosine(mean_vec, i)
        else:
            print(
                "distance type not known: enter either of eucos," +
                " euclidean or cosine")
    return query_distance_list
