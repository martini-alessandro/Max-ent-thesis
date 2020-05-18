# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:19:09 2020

@author: Workplace
"""
import pandas as pd
import numpy as np 
import scipy.stats 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.linalg 



def gaussianSpectrum(w):
    g1 = scipy.stats.norm.pdf(w, loc = 0) 
    g2 = scipy.stats.norm.pdf(np.flip(-w))
    conc = np.concatenate((g1, g2))
    return conc

def back_to_autocorr(w, time):
    spec = gaussianSpectrum(w)
    gamma = np.real(np.fft.ifft(spec))[: len(w)]  
    n = len(gamma)
    gamma /= n 
    c = interp1d(time, gamma)
    return c

def constructCovariance(c, time, ConstructAutocorr = False, w = None): 
    if ConstructAutocorr and w == None: 
        raise ValueError('w vector is needed to construct the autocovariance')
    elif ConstructAutocorr:
        c = back_to_autocorr(w, time)
    autocorr = c(time)
    return scipy.linalg.toeplitz(autocorr)

def gauss_noise(cov): 
    mean = np.zeros(cov.shape[0])
    return np.random.multivariate_normal(mean, cov)


if __name__ == "__main__": 
    ax = np.linspace(0,5,10000)
    time = np.linspace(0,5, 10000)
    #n = np.real(np.fft.ifft(function(ax))).shape[0]
    #plt.plot(ax, spectrum(ax))
    #n = len(ax)
    #gamma = np.real(np.fft.ifft(gaussianSpectrum(ax)))[: n // 2]
    #plt.xlim(0,.1)