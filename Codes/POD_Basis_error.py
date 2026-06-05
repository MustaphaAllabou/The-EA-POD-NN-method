#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:20:27 2024

@author: allabou
"""
import numpy as np
import os
import h5py
import numpy.linalg as npl
import scipy
import matplotlib.pyplot as plt
from math import atan
from numpy.linalg import eig
import pandas as pd
from scipy.linalg import eigh
import time
from datetime import datetime
from POD_Basis import PODProcessor

class PODProcessor_error: 
    def __init__(self,epsilon_er=None):
        self.epsilon_er = epsilon_er
        self.Snapshots = None
        self.Eigen_val = None
        self.Eigen_vec = None
        self.Ber = None
        self.s_eigen = None


    def compute_eigenvalues_er(self,Snapshots,Nrb,Brb):
        print('Nrb is equal to',Nrb)
        print('Brb is of shape',Brb.shape)
        #Er = Snapshots - np.dot(Brb, np.dot(Brb.transpose(), Snapshots))
        Er = Snapshots - np.dot(Brb, np.dot(Brb.transpose(), Snapshots))
        C = np.dot(Er.transpose(), Er)
        print("cond of C=", npl.cond(C, 2))
        start_time = datetime.now()
        self.Eigen_val, self.Eigen_vec = np.linalg.eigh(C)
        a = np.argsort(-self.Eigen_val)
        self.Eigen_val = self.Eigen_val[a]
        self.Eigen_vec = self.Eigen_vec[:, a]
        Xi = np.zeros((len(Snapshots),len(self.Eigen_vec)))
        for i in range(len(self.Eigen_vec)):
            Xi[:,i] = (1/(np.sqrt(self.Eigen_val[i])))*np.dot(Snapshots,self.Eigen_vec[:,i])
        self.Eigen_vec = Xi
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        np.save('Eigen_values_error', self.Eigen_val)
        np.save('Er',Er)

    def plot_eigenvalues_er(self):
        a = np.argsort(-self.Eigen_val)
        self.Eigen_val = self.Eigen_val[a]
        Eigen_vec = self.Eigen_vec[:, a]

        Eigen_val_normalized = self.normalize_eigenvalues_er()

        Decay = np.arange(len(self.Eigen_val))
        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        ax.plot(Decay, Eigen_val_normalized, label='Normalized errors eigenvalues', color='r')
        ax.set_xticks(Decay)
        ax.set_xticklabels([r'$\lambda_{%i}$' % (i + 1) for i in Decay], fontsize=20)
        plt.xscale("log")
        plt.yscale("log")
        width = 0.5
        p = plt.bar(Decay, Eigen_val_normalized, width, color='b')
        plt.legend(fontsize=16)
        ax.legend(fontsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.show()
        
    def normalize_eigenvalues_er(self):
        if self.s_eigen is None:
            self.s_eigen = np.sum(self.Eigen_val)
            print('s_eigen =', self.s_eigen)
        Eigen_val_normalized = (self.Eigen_val) / self.s_eigen
        return Eigen_val_normalized

    def compute_number_of_modes_er(self, epsilon_er):
        Ner = self.energy_number_er(epsilon_er)
        print('This corresponds to Ner = ', Ner)
        #self.truncate_reduced_basis_er(Ner)
        return Ner
    
    def energy_number_er(self, epsilon_er):
        Eigen_val_normalized = self.normalize_eigenvalues_er()
        s1 = np.sum(Eigen_val_normalized)
        s = 0.
        index_min = 0
        for i in range(len(Eigen_val_normalized)):
            if s < s1 * (1 - epsilon_er):
                s += Eigen_val_normalized[i]
                index_min += 1
        return index_min

    def truncate_reduced_basis_er(self, Ner):
        if self.Ber is None:
            self.Ber = self.Eigen_vec[:, 0:Ner]
            print("Ber shape is", self.Ber.shape)
            np.save('Ber', self.Ber)
        return self.Ber

    def compute_error_estimation_er(self,Snapshots):
        Uh = np.zeros(len(Snapshots.transpose()))
        S = 0.

        for m in range(len(Snapshots.transpose()[0])):
            Uh = Snapshots.transpose()[:, m]
            Urb = np.dot(self.Ber, np.dot(self.Ber.transpose(), Uh))
            Diff = Uh - Urb
            S += np.dot(Diff.transpose(), Diff)

        Discarded_eigenvalues = np.sum(self.Eigen_val[len(self.Ber):])
        print("The error estimation by POD method for error matrix is", abs(S - Discarded_eigenvalues) / abs(self.s_eigen))


# pod_params_er = {
#     'epsilon_er': 1e-2,
# }       


# pod_params = {
#     'M': 5,
#     'Nt': 500,
#     'Nh': 7267,
#     'epsilon_POD': 1e-2,
#     'data_directory': '/home-local/allabou/Bureau/These/Code/Code_reduction/model_outputs/'
# } 
      
# # Compute the RB matrix Brb:
# pod_processor = PODProcessor(**pod_params)
   
# # Read the hdf5 files and construct the snapshots matrix
# Snapshots = pod_processor.read_data_files()
# #Snapshots = np.load('Snapshots.npy')
# print('Snapshots shaps is',Snapshots.shape)
    
# # Comute the correlation matrix C
# pod_processor.compute_eigenvalues()
# # Compute and plot the eigenvalues and eigenvectors of C
# pod_processor.plot_eigenvalues() 
# # Truncate the first Nrb modes
# # Truncate the first Nrb modes
# Nrb = pod_processor.compute_number_of_modes(pod_params['epsilon_POD'])
# #Nrb = 5
# # print('Nrb=',Nrb)

# # Get the RB matrix
# Brb = pod_processor.truncate_reduced_basis(Nrb)
# # Brb = np.load('Brb.npy')   
# # Nrb = len(Brb[0]) 
# # # Verify the error estimtion of POD method
# # pod_processor.compute_error_estimation()  # Uncomment this line if needed

# #############################################################################################################"""""
# # The error part
# #############################################################################################################

# # Compute the RB matrix Brb:
# pod_processor_er = PODProcessor_error(**pod_params_er)
# # Comute the correlation matrix C
# pod_processor_er.compute_eigenvalues_er(Snapshots,Nrb,Brb)
# # Compute and plot the eigenvalues and eigenvectors of C
# pod_processor_er.plot_eigenvalues_er() 
# # Truncate the first Nrb modes
# Ner = pod_processor_er.compute_number_of_modes_er(pod_params1['epsilon_er'])

# # Get the RB matrix
# Ber = pod_processor_er.truncate_reduced_basis_er(Ner)