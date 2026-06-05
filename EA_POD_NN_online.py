#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:56:37 2024

@author: allabou
"""
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.datasets import mnist
import tensorflow.keras.preprocessing.image as kpi
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kloss
import tensorflow.keras.regularizers as kr
import tensorflow.keras.backend as K
import tensorflow.keras.utils as ku
from tensorflow.keras import callbacks
import scipy as sc
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import sys,os
import random 
import pandas as pd
import sklearn.utils
import time
from EA_POD_NN_offline import POD_NN_METHOD
# Verbosity
fit_verbosity = 1

class POD_NN_PREDICTION:
    def __init__(self, model, model_er, Brb, Brb_err, max_P, min_P, Urb_POD_max, Urb_POD_min, Urb_POD_er_max, Urb_POD_er_min,Nh=7267):
        self.model = model
        self.model_er = model_er
        self.Brb = Brb
        self.Brb_err = Brb_err
        self.max_P = max_P
        self.min_P = min_P
        self.Urb_POD_max = Urb_POD_max
        self.Urb_POD_min = Urb_POD_min
        self.Urb_POD_er_max = Urb_POD_er_max
        self.Urb_POD_er_min = Urb_POD_er_min
        self.Nh = Nh
    
    def normalize_data(self, data):
        data_max = np.max(data)
        data_min = np.min(data)
        normalized_data = (data - data_min) / (data_max - data_min)
        return normalized_data, data_max, data_min
    
    def inverse_normalize_data(self, normalized_data,data_min,data_max):
        """
        Inverse normalize the input data that was previously normalized using Min-Max scaling.
           
        Returns:
        - original_data: The data after inverse normalization.
        """
        original_data = normalized_data * (data_max - data_min) + data_min
        return original_data
    
    def predict_solution(self,my_nn, Param, Time_simu,Nh):
        Uh_POD_NN_All = np.zeros((Nh, len(Time_simu)))

        for i in range(len(Time_simu)):
            time_simu = Time_simu[i]
            param = np.array([[Param[0], Param[1], Param[2], Param[3], time_simu]])

            self.normalize_data(param)

            urb = my_nn.model.predict(param)
            urb_err = my_nn.model_er.predict(param)

            self.inverse_normalize_data(urb, self.Urb_POD_max, self.Urb_POD_min)
            self.inverse_normalize_data(urb_err, self.Urb_POD_er_max, self.Urb_POD_er_min)

            Uh_POD_NN = np.dot(self.Brb, urb.transpose())
            Uh_POD_NN_err = np.dot(self.Brb_err, urb_err.transpose())
            Uh_POD_NN = Uh_POD_NN + Uh_POD_NN_err

            Uh_POD_NN = Uh_POD_NN.reshape(1, len(Uh_POD_NN))
            #print('Uh_POD_NN shape',Uh_POD_NN.shape)
            Uh_POD_NN_All[:, i] = Uh_POD_NN.flatten()
        print('Uh_POD_NN shape',Uh_POD_NN.shape)
        h_POD_NN = Uh_POD_NN_All[:self.Nh,:];u_POD_NN = Uh_POD_NN_All[self.Nh:2*self.Nh,:];v_POD_NN = Uh_POD_NN_All[2*self.Nh:,:]
        norm_vel = np.sqrt(u_POD_NN**2 + v_POD_NN**2)
        np.save('Uh_EA_POD_NN', Uh_POD_NN_All)
        return h_POD_NN, norm_vel

