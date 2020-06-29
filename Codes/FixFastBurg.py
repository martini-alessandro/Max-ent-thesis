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


if __name__ == '__main__': 
    #Generate array of noise
    dt = 1 / 1024
    T = 8
    fmax = 1 / (2 * dt)
    f, ps = np.loadtxt('..{}LIGO-P1200087-v18-AdV_DESIGN_psd.dat'.format(os.sep), unpack = True)
    
    time, time_series, freq, freqSeries, psd = fr.generate_data(f, ps, T, 0,
                                                                1 / dt, 10, 
                                                                fmax)
    N = len(time_series)
    M = int(2*N / np.log(2*N))
    
    #Compute variables for burg Methods: fast with different dr and standard Burg
    lenght = 10
    p, ak, ks, gs, Drs, rs = Fb2.Fastburg(time_series, lenght, dr = 2)
    p2, a2, ks2, gs2, Drs2, rs2 = Fb2.Fastburg(time_series, lenght, dr = 1)
    bp, bak, bk = sa.burgMethod(time_series, lenght)
    p, p2, bp = np.array(p), np.array(p2), np.array(bp)
    
    #Compare results for p 
    ppFast = p / bp
    ppOut = p2 / bp
    one = np.ones(lenght)
    
    #plot the ratios and ratio = 1 line 
    plt.plot(one, color = 'r', linestyle = '--')
    plt.plot(ppFast, label = 'Fast')
    plt.plot(ppOut, label = 'Outer')
    plt.legend()

