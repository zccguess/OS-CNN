# import os, sys, pickle, glob
# import os.path as path
import sys
import argparse
import time
# import scipy.spatial.distance as spd
import scipy
import scipy as sp
from scipy.io import loadmat
import numpy as np
from utils.openmax_utils import compute_distance, getlabellist
from utils.evt_fitting import weibull_tailfitting, query_weibull
import libmr


# ---------------------------------------------------------------------------------
# params and configuratoins
NCHANNELS = 1
NCLASSES = 7#C0
# ---------------------------------------------------------------------------------


def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    prob_scores, prob_unknowns = [], []
    open = []
    modified_scores = np.zeros((len(openmax_fc8), len(openmax_fc8[0])+1))
    for channel in range(len(modified_scores)):
        channel_scores, channel_unknowns = [], []
        for category in range(NCLASSES):
            know = openmax_fc8[channel, category]
            channel_scores += [np.exp(know)]
        sum_un = np.sum(openmax_score_u[channel, :])
        channel_unknown_scores = np.exp(sum_un)#softmax
        total_denominator = np.sum(channel_scores) + channel_unknown_scores
    #######################softmax#########################################
        inf = float("inf")
        # nan = float("nan")
        prob_scores = []
        for i in range(len(modified_scores[0])-1):
            if total_denominator == inf:
                if channel_scores[i] != inf :
                    prob_scores +=[0]
                else:
                    prob_scores += [channel_scores[i] / total_denominator]
            else:
                prob_scores += [channel_scores[i] / total_denominator]
        prob_unknowns = [
            channel_unknown_scores/total_denominator]
        #############################################
        prob_scores = sp.asarray(prob_scores)
        prob_scores = prob_scores.reshape(1,-1)
        prob_unknowns = sp.asarray(prob_unknowns)
        scores = sp.mean(prob_scores, axis=0)
        unknowns = sp.mean(prob_unknowns, axis=0)
        modified_scores[channel] += (scores.tolist() + [unknowns])
        open.append(np.argmax(modified_scores[channel]))
        assert len(modified_scores[channel]) == len(modified_scores[0])#J2-J4
    return modified_scores
# ---------------------------------------------------------------------------------
def recalibrate_scores(weibull_model, labellist, imgarr,
                       layer='fc8', alpharank=7, distance_type='eucos'):
    #######################################################################
    imglayer = imgarr[layer]
    ####################################################################
    ranked_list = np.argsort(-imglayer)
    alpha_weights = [
             ((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]####
    ranked_alpha = sp.zeros((len(imglayer), len(labellist)))  #p3-5ȱ2
    for i in range(len(ranked_list)):
        for j in range(len(alpha_weights)):
            ranked_alpha[i][ranked_list[i][j]] = alpha_weights[j]
    openmax_fc8, openmax_score_u = [], []
    for channel in range(NCHANNELS):
        channel_scores = imglayer[channel:]
        openmax_fc8_channel = np.zeros((len(imglayer), len(labellist))) # p3-5
        openmax_fc8_unknown = np.zeros((len(imglayer), len(labellist)))
        wscore_list = np.zeros((len(imglayer), len(labellist)))
        w_list = np.zeros((len(imglayer), len(labellist)))
        w2_list = np.zeros((len(imglayer), len(labellist)))
        channel_distance_list=[]
        channel_undistance_list=[]
        for categoryid in range(NCLASSES):
            category_weibull = query_weibull(
                labellist[categoryid],
                weibull_model, distance_type=distance_type)
            channel_distance = compute_distance(
                channel_scores, channel, category_weibull[0],distance_type=distance_type)
            channel_distance_list.append(channel_distance[categoryid*100:(categoryid+1)*100])
            channel_undistance_list.append(channel_distance[500:600])
        #####################################################################################xiu
            for i in range(len(channel_distance)):
                wscore = category_weibull[2][channel].w_score(channel_distance[i])
                wscore_list[i][categoryid] += [wscore]
                w_list[i][categoryid] += [1 - wscore]
                w2_list[i][categoryid] += [1 - wscore * ranked_alpha[i,:][categoryid]]

                modified_fc8_score = np.exp(channel_scores[i,:][categoryid]) * (1 - wscore * ranked_alpha[i,:][categoryid])
                openmax_fc8_channel[i][categoryid]+=[modified_fc8_score]
                openmax_fc8_unknown[i][categoryid] += [np.exp(channel_scores[i,:][categoryid]) - modified_fc8_score]
        channel_distance_array = np.array(channel_distance_list)
        channel_undistance_array = np.array(channel_undistance_list)
        scipy.io.savemat('distance.mat', mdict={'distance': channel_distance_array})
        scipy.io.savemat('undistance.mat', mdict={'distance': channel_undistance_array})
        scipy.io.savemat('wscore.mat', mdict={'wscore': wscore_list})
        scipy.io.savemat('w.mat', mdict={'w': w_list})
        scipy.io.savemat('k.mat', mdict={'k': openmax_fc8_channel})
        scipy.io.savemat('un.mat', mdict={'un': openmax_fc8_unknown})
    openmax_probab = computeOpenMaxProbability(openmax_fc8_channel, openmax_fc8_unknown)
    softmax_probab = imgarr['scores']
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab)