# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:56:05 2020

@author: Workplace
"""

import ConstrucFreqNoise
from MESAAlgorithm import MESA 
import numpy as np
import matplotlib.pyplot as plt 
import os
import statsmodels.tsa as tsa
import scipy.stats

if __name__ == '__main__':
    ar = np.array([1, '...'])
    tsa.arima_process.arma_generate_sample(ar, np.array(), \
                                           scale = 1, nsample = 100000)
    #Originating a ARMA(p, q). MA order has to include 0'th order, so, for a pure
    #AR process, arg should be np.array([1]). scale is te scale of univariate normal
    #from which error is extrapolated. 
    f = np.linspace(0.1, 10, 1000)
    #ps = .7 * scipy.stats.norm.pdf(f, 3, .5) + .3 * scipy.stats.norm.pdf(f, .3, .1)
    ps = 1 / f + .7 * scipy.stats.norm.pdf(f, 3, .5)
    dt = 1 / 10
    Ny = 1 / (2 * dt)
    
    # f, ps = np.loadtxt('..{}LIGO-P1200087-v18-AdV_DESIGN_psd.dat'.format(os.sep), unpack = True)
    # #dt = 1 / (2 * f[-1])
    # dt = 1 / 1024
    # Ny = 1 / (2 * dt)
    # N = int(32 / dt)
    time, noise, freq, fnoise, PSD = ConstrucFreqNoise.generate_data(f, ps, T= 5000,
                                                                     sampling_rate = 1 / dt,
                                                                      fmin = 10, fmax = Ny)
    # #fNoise, noise = generate_noise_roughly(f, ps)
    # # plt.loglog(f, ps)
    # # plt.loglog(freq, PSD)
    
    M = MESA(noise)
    P, ak = M.solve(optimisation_method = 'OBD', method = 'Fast')
    f1 = np.linspace(10, f[-1], len(f))
    plt.plot(freq, M.spectrum(dt, freq))
    plt.plot(f, ps, color = 'r', linestyle = '--')