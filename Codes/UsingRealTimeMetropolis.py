# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:29:28 2020

@author: Workplace
"""

import MetropolisModified as MM
import matplotlib.pyplot as plt
import numpy as np 


if __name__ == '__main__': 
    np.random.seed(1)
    #Generate random parameters 
    amplitude, frequency, phase = np.random.uniform([23,1,0], [24,10,2 * np.pi])
    print('Parameters values are\nAmplitude: {},\nFrequency: {}\
          \nPhase: {}'.format(amplitude, frequency, phase))
          
    #Define parameter estimators
    A = MM.Parameter('uniform', 23, 24, .0)
    A.value = amplitude
    F = MM.Parameter('uniform', 1, 10, .001)
    F.value = np.random.normal(frequency, .5)
    print(F.value)
    P = MM.Parameter('uniform', 0, 2 * np.pi, 0)
    P.value = phase
    
    #Create Data
    NumberOfData = 1000
    dt = 1 / 40
    x = np.arange(0, NumberOfData) * dt
    T = NumberOfData  * dt
    error = np.random.normal(0, .8 * amplitude, NumberOfData)
    y = amplitude * np.sin(frequency * x + phase) + error
    D = MM.Data(x, y)
    
    #Define distributions and likelihood
    p = MM.Prior(A, F, P)
    L = MM.Likelihood(D, 1 / T , 'sin', A, F, P)
    Post = MM.Posterior(L, p)
    
    #Define Sampling 
    Ny = 1 / (2 * dt)
    freq = np.arange(0, NumberOfData / 2 + 1) / T 
    # freq = np.linspace(0, Ny, NumberOfData // 2 + 1)
    #Metropolis
    jumps = [0, .005, .05, .5, 1]
    M = MM.Metropolis(Post, 10000)
    fig, ax = plt.subplots()
    for i in range(len(jumps)): 
        print(( i + 1 ) / len(jumps))
        F.value = np.random.normal(frequency, jumps[i])
        samples, specta = M.realTimeEvaluate(dt = dt, f = freq)
        ax.plot(samples[:,1], '.', label = i)
        
    