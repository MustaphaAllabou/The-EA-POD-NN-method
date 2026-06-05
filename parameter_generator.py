#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:47:02 2024

@author: allabou
"""
import numpy as np
import matplotlib.pyplot as plt
import subprocess

class HydrographGenerator:
    def __init__(self, n=48, q1_max=7.0, q2_max=2.0, step=172800. / 48):
        self.n = n
        self.Q1 = np.zeros(n)
        self.Q2 = np.zeros(n)
        self.Q1[:13] = q1_max
        self.Q2[:13] = q2_max
        self.qmin1 = np.min(self.Q1)
        self.qmin2 = np.min(self.Q2)
        self.qmax1 = np.max(self.Q1)
        self.qmax2 = np.max(self.Q2)
        self.step = step
        self.Time = np.arange(start=3600, stop=172800. + 3600., step=step)
        self.Time_simu = np.arange(start=0., stop=216000. + 10800., step=10800.)
        self.Time_simu = self.Time_simu[12:]
        np.save('Time_simu', self.Time_simu)

    def f(self, t, tmax, qmax, tmente, qmin, tdescente):
        if tmente <= t <= tmax:
            a = (qmax - qmin) / (tmax - tmente)
        elif tmax <= t <= tdescente:
            a = (qmin - qmax) / (tdescente - tmax)
        else:
            return 0  # Assuming 0 if t is outside the specified ranges
        b = qmax - a * tmax
        return a * t + b

    def generate_parameter_space(self, mu1_range, mu2_range, mu3_range, mu4_range, constraint):
        parameter_space = []

        for mu1 in mu1_range:
            for mu3 in mu3_range:
                if abs(mu1 - mu3) / abs(mu1) <= constraint:
                    for mu2 in mu2_range:
                        for mu4 in mu4_range:
                            parameter_space.append((mu1, mu2, mu3, mu4))

        return parameter_space

    def check_constraint(self, param, constraint):
        if abs(param[0] - param[2]) / abs(param[0]) <= constraint:
            print('The point is good with constraint:', abs(param[0] - param[2]) / abs(param[0]))
        else:
            print('The point is not good because the constraint is:',
                  abs(param[0] - param[2]) / abs(param[0]))
    def plot_parameter_space(self, echantillons, param, tmontee1, tmontee2):
        Tmax1 = echantillons[:, 0] - tmontee1
        Tmax2 = echantillons[:, 2] - tmontee2
        Qmax1 = echantillons[:, 1]
        Qmax2 = echantillons[:, 3]

        #Tmax1 = Tmax1 / 3600.
        #Tmax2 = Tmax2 / 3600.

        fig1, ax1 = plt.subplots()
        ax1.scatter(Tmax1, Tmax2, s=5, color='b', marker='o')
        ax1.set_xlabel('$t_{peak,1}$ ($h$)', fontsize=14)
        ax1.set_ylabel('$t_{peak,2}$ ($h$)', fontsize=12)
        #ax1.scatter(param[0] / 3600., param[2] / 3600., color='red', marker='o')
        #fig1.savefig('The_parameter_space_t1vst2_phased.png')

        fig2, ax2 = plt.subplots()
        ax2.scatter(Tmax1, Qmax1, s=5, color='b', marker='o')
        ax2.set_xlabel('$t_{peak,1}$ ($h$)', fontsize=14)
        ax2.set_ylabel('Discharge $Q_{max,1}$ ($m^3.s^{-1}$)', fontsize=12)
        #ax2.scatter(param[0] / 3600., param[1], color='red', marker='o')
        #fig2.savefig('The_parameter_space_t1vsQ1_phased.png')

        fig3, ax3 = plt.subplots()
        ax3.scatter(Tmax2, Qmax2, s=5, color='b', marker='o')
        ax3.set_xlabel('$t_{peak,1}$ ($h$)', fontsize=14)
        ax3.set_ylabel('Discharge $Q_{max,2}$ ($m^3.s^{-1}$)', fontsize=12)
        #ax3.scatter(param[2] / 3600., param[3], color='red', marker='o')
        #fig3.savefig('The_parameter_space_t2vsQ2_phased.png')
        plt.show()
        
    def generate_hydrographs(self, echantillons, tmontee1, tmontee2):
        Qin1 = np.zeros((self.n, len(echantillons)))
        Qin2 = np.zeros((self.n, len(echantillons)))
        
        for m in range(len(echantillons)):
            tmax1 = echantillons[m, 0] 
            tmax2 = echantillons[m, 2] 
            tdescente1 = 2 * tmax1 - tmontee1
            tdescente2 = 2 * tmax2 - tmontee2
            Q1 = np.full(self.n, 7.0)
            Q2 = np.full(self.n, 2.0)
            qmax1 = echantillons[m, 1]
            qmax2 = echantillons[m, 3]

            for i in range(self.n):
                if tmontee1 <= self.Time[i] <= tdescente1:
                    Q1[i] = self.f(self.Time[i], tmax1, qmax1, tmontee1, self.qmin1, tdescente1)
                if tmontee2 <= self.Time[i] <= tdescente2:
                    Q2[i] = self.f(self.Time[i], tmax2, qmax2, tmontee2, self.qmin2, tdescente2)
                    
            plt.plot(self.Time, Q1, label='Q1 real')
            plt.plot(self.Time, Q2, label='Q2 real')
            plt.title('The inflow for Qmax1=' + str(qmax1) + '$m^3.s^{-1}$ and Qmax2=' + str(qmax2) + ' (s)')
            plt.xlabel(r'Time ($s$)')
            plt.ylabel(r'Discharge ($m^3.s^{-1}$)')
            plt.legend()
            plt.show()    

            Qin1[:, m] = Q1
            Qin2[:, m] = Q2
            np.save('Qin1',Qin1);np.save('Qin2',Qin2)

        return Qin1, Qin2
    
    def generate_hydrograph_online(self, Q1, Q2, Time, tmontee, param):
        Q1 = Q1 + 7.; Q2 = Q2 + 2. 
        tmax1 = param[0] + tmontee; tmax2 = param[2] + tmontee
        qmax1 = param[1]; qmax2 = param[3]
        tdescente1 = 2 * tmax1 - tmontee; tdescente2 = 2 * tmax2 - tmontee
        for i in range(len(Q1)):
            if (Time[i] >= tmontee and Time[i] <= tdescente1):
                Q1[i] = self.f(Time[i], tmax1, qmax1, tmontee, self.qmin1, tdescente1)
            if (Time[i] >= tmontee and Time[i] <= tdescente2):
                Q2[i] = self.f(Time[i], tmax2, qmax2, tmontee, self.qmin2, tdescente2)

        plt.plot(Time, Q1, label='Q1 real')
        plt.plot(Time, Q2, label='Q2 real')
        plt.title('The inflow for Qmax1=' + str(qmax1) + '$m^3.s^{-1}$ and Qmax2=' + str(qmax2) + ' (s)')
        plt.xlabel(r'Time ($s$)')
        plt.ylabel(r'Discharge ($m^3.s^{-1}$)')
        plt.legend()
        plt.show()
        
        nameHydro         ='hydrograph.txt'
        linspace=len(Time)
        print("Number of elements in Time:", len(Time))
        print("Number of elements in Flow1:", len(Q1))
        file=open(nameHydro,'w+')
        file.write('#comment\n')
        file.write('#comment\n')
        file.write('#comment\n')
        file.write(str((3))+'\n')
        file.write('#comment\n')
        file.write('#comment\n')
        file.write('#comment\n')
        file.write(str((linspace))+'\n')
        for i in range(linspace):
            file.write(str(Time[i]) + ' ' +str(0.0)+'\n')
        file.write('#comment\n')
        file.write('#comment\n')
        file.write('#comment\n')
        file.write(str((linspace))+'\n')
        for i in range(linspace):
            file.write(str(Time[i]) + ' ' +str(Q1[i])+'\n')
        file.write('#comment\n')
        file.write('#comment\n')
        file.write('#comment\n')
        file.write(str((linspace))+'\n')
        for i in range(linspace):
            file.write(str(Time[i]) + ' ' +str(Q2[i])+'\n')
        file.write('#comment\n')
        file.close()
        
        np.save('Q_1', Q1)
        np.save('Q_2', Q2)
        subprocess.run(['cp', nameHydro, '/home-local/allabou/Bureau/dasshydro-main/cases/dev_case/Aude/built_bin/'])
    

    def generate_pod_nn_parameters(self, echantillons,tmontee1,tmontee2):
        POD_NN_Parameters = []

        for j in range(len(echantillons)):
            for k in range(len(self.Time_simu)):
                tmax1 = echantillons[j, 0] + tmontee1
                qmax1 = echantillons[j, 1]
                tmax2 = echantillons[j, 2] + tmontee2
                qmax2 = echantillons[j, 3]
                time_simu_k = self.Time_simu[k]

                POD_NN_Parameters.append([tmax1 - tmontee1, qmax1, tmax2 - tmontee2, qmax2, time_simu_k])
        POD_NN_Parameters = np.array(POD_NN_Parameters)
        np.save('POD_NN_Parameters',POD_NN_Parameters)

        return POD_NN_Parameters





