# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:41:00 2020

@author: Workplace
"""
import ConstrucFreqNoise as fr 
import SolarSpotsAnalysis as sa
import FastBurg2 as Fb2
import numpy as np
import matplotlib.pyplot as plt
import os
from MESAAlgorithm import MESA


if __name__ == '__main__': 
    #Generate array of noise
    dt = 1 / 2048
    T = 12
    fmax = 1 / (2 * dt)
    f, ps = np.loadtxt('..{}LIGO-P1200087-v18-AdV_DESIGN_psd.dat'.format(os.sep), unpack = True)
    
    time, time_series, freq, freqSeries, psd = fr.generate_data(f, ps, T, 0,
                                                                1 / dt, 10, 
                                                                fmax)
    M = MESA(time_series)
    P, ak = M.solve(method = "Fast", optimisation_method = "FPE")
    print('fast :',P, len(ak))
    plt.loglog(f, M.spectrum(dt,f))
    P, ak = M.solve(method = "Standard", optimisation_method = "FPE")
    print('slow :',P, len(ak))
#    exit()
    plt.loglog(f, M.spectrum(dt,f), '--')
    plt.show()
   
