# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:41:00 2020

@author: Workplace
"""
import ConstrucFreqNoise as fr 
import SolarSpotsAnalysis as sa
import FastBurg2 as Fb2n
import numpy as np
import matplotlib.pyplot as plt
import os
from MESAAlgorithm import MESA
#from colorednoise import powerlaw_psd_gaussian

if __name__ == '__main__': 
    #Generate array of noise
    dt = 1 / 4096
    T = 8
    # fmax = 1 / (2 * dt)
    f, ps = np.loadtxt('..{}LIGO-P1200087-v18-AdV_DESIGN_psd.dat'.format(os.sep), unpack = True)
    #f = np.linspace(0, fmax, 20000)
   

    times, t, freq, freqSeries, psd = fr.generate_data(f, ps, T, 0, 1 / dt)
                                                                
    N = MESA(t)
    P, ak, obptim = N.solve(method = "Fast", optimisation_method = "FPE")
    plt.loglog(freq, N.spectrum(dt,freq), color = 'k', linestyle = '--', label = 'Fast')
    plt.loglog(f, ps)
    # P, ak = N.solve(method = "Standard", optimisation_method = "CAT")
    # plt.loglog(f, N.spectrum(dt,f), linestyle = '-.', label = 'Standard') 
    # print('slow :',P, len(ak))
    # plt.loglog(f, M.spectrum(dt,f), color = 'k', label = 'From time_series')
    # plt.legend() 
   
