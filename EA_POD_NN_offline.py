#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:48:19 2024

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
from sklearn.preprocessing import MinMaxScaler
# Verbosity
fit_verbosity = 1

class POD_NN_METHOD:
    def __init__(self,input_dim, Nrb, Ner, Neurons, Neurons_er, epochs, batch_size, lr, alpha, alpha1):
        self.input_dim = input_dim
        self.Nrb = Nrb
        self.Ner = Ner
        self.Neurons = Neurons
        self.Neurons_er = Neurons_er
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.alpha1 = alpha1
        
        # ... le reste de votre constructeur ...

        self.model = self.build_model(Neurons=self.Neurons)
        self.model_er = self.build_model_er(Neurons_er=self.Neurons_er)
        self.max_P = None
        self.min_P = None
        self.Urb_POD_max1 = None
        self.Urb_POD_min1 = None
        self.Brb = None
        self.Ber = None
        self.Er = None
        self.scaler = MinMaxScaler()
        
        # Check for GPU and set memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Vous avez au moins un GPU
            print("Using GPU:", gpus[0])
            try:
                # Limit TensorFlow to use the first GPU
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                
                # Specify dynamic memory usage for the GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                print("Using GPU:", gpus[0])
            except RuntimeError as e:
                print(e)
        else:
            # Aucun GPU disponible, TensorFlow utilisera le CPU
            print("No GPU detected. Using CPU.")



    def build_model(self, Neurons):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(Neurons, input_dim=5, activation='relu'),  # Updated input_dim to 5
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(Neurons, activation='relu'),
            tf.keras.layers.Dense(self.Nrb)
        ])
        model.summary()
        return model
    
    def build_model_er(self, Neurons_er):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(Neurons_er, input_dim=5, activation='relu'),  # Updated input_dim to 5
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(Neurons_er, activation='relu'),
            tf.keras.layers.Dense(self.Ner)
        ])
        model.summary()
        return model
    
    def normalize_data(self, data):
        data_max = np.max(data)
        data_min = np.min(data)
        normalized_data = (data - data_min) / (data_max - data_min)
        return normalized_data, data_max, data_min
    
    def inverse_normalize_data(normalized_data,data_max,data_min):
        """
        Inverse normalize the input data that was previously normalized using Min-Max scaling.
           
        Returns:
        - original_data: The data after inverse normalization.
        """
        original_data = normalized_data * (data_max - data_min) + data_min
        return original_data

    def scaling(self, S, S_max, S_min):
        S[...] = (S - S_min) / (S_max - S_min)

    def inverse_scaling(self, S, S_max, S_min):
        S[...] = (S_max - S_min) * S + S_min

    def load_data(self):
        # Load your data here (S, Brb, P, etc.)
        self.S = np.load('Snapshots.npy')
        self.Brb = np.load('Brb.npy')
        self.P = np.load('POD_NN_Parameters.npy')
        self.P = self.P[:89,:]
        self.Er = np.load('Er.npy')
        self.Ber = np.load('Ber.npy')
        
        #
        # Compute the reduced data
        self.Urb_POD = np.dot(self.Brb.transpose(), self.S).transpose()
        self.Urb_POD_er = np.dot(self.Ber.transpose(), self.Er).transpose()
        
        # Or sequentialy
        
        print("len Urb_POD_train=", self.Urb_POD.shape)
        print("len Urb_POD_train_er=", self.Urb_POD_er.shape)
        print("len P=", self.P.shape)

    def split_data(self):
        N = len(self.P)
        alpha = self.alpha
        alpha1 = self.alpha1
        N_tr = int(N * (1 - (alpha + alpha1)))
        N_test = int(N * (1 - alpha1))
        
        # Compute the train data
        self.Urb_POD_train = self.Urb_POD[:N_tr, :]
        self.Urb_POD_train_er = self.Urb_POD_er[:N_tr, :]
        
        # Compute the validation data
        self.Urb_POD_val = self.Urb_POD[N_tr:N_test, :]
        self.Urb_POD_val_er = self.Urb_POD_er[N_tr:N_test, :]
        
        # Compute the test data
        self.Urb_POD_test = self.Urb_POD[N_test:N, :]
        self.Urb_POD_test_er = self.Urb_POD_er[N_test:N, :]
        
        self.P_train = self.P[:N_tr]
        self.P_val = self.P[N_tr:N_test]
        self.P_test = self.P[N_test:N]
        
        print("len P_train=", self.P_train.shape)
        print("len Urb_POD_train=", self.Urb_POD_train.shape)
        print("len Urb_POD_train_er=", self.Urb_POD_train_er.shape)
        print("len P_val=", self.P_val.shape)
        print("len Urb_POD_val=", self.Urb_POD_val.shape)
        print("len Urb_POD_val_er=", self.Urb_POD_val_er.shape)
        print("len P_test=", self.P_test.shape)
        print("len Urb_POD_test=", self.Urb_POD_test.shape)
        print("len Urb_POD_test_er=", self.Urb_POD_test_er.shape)






    def train_models(self, fit_verbosity=1):
        optim = tf.keras.optimizers.Adam(lr=self.lr)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)     
        optim_er = tf.keras.optimizers.Adam(lr=self.lr)  # Nouvel optimiseur pour le deuxième modèle
    
        with tf.device('/GPU:0'):   
            self.model.compile(optimizer=optim, loss='mse', metrics=['accuracy'])
              # Utiliser le nouvel optimiseur
        
            # Train the first model
            self.history = self.model.fit(self.P_train, self.Urb_POD_train,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      verbose=fit_verbosity,
                                      validation_data=(self.P_test, self.Urb_POD_test))#,
                                     # callbacks=[callback])
        
        with tf.device('/GPU:0'):
            self.model_er.compile(optimizer=optim_er, loss='mse', metrics=['accuracy'])
            # Train the second model
            self.history_er = self.model_er.fit(self.P_train, self.Urb_POD_train_er,
                                            epochs=self.epochs,
                                            batch_size=self.batch_size,
                                            verbose=fit_verbosity,
                                            validation_data=(self.P_test, self.Urb_POD_test_er))#,
                                          #  callbacks=[callback])
    
        return self.history, self.history_er





    def evaluate_model(self):
        score = self.model.evaluate(self.P_val, self.Urb_POD_val)
        score_er = self.model_er.evaluate(self.P_val, self.Urb_POD_val_er)

    def plot_loss(self, history, history_er):
        plt.plot(history.history['loss'], "b", label='train_loss')
        plt.plot(history.history['val_loss'], "r", label='val_loss')
        plt.plot(history_er.history['loss'], "g", label='train_loss_er')
        plt.plot(history_er.history['val_loss'], "y", label='val_loss_er')
        
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(loc='upper right')
        plt.yscale("log")
        
        plt.savefig("NN_training_loss.png")
        plt.show()




    def save_model(self, filename1,filename2):
        self.model.save(filename1)
        self.model_er.save(filename2)
