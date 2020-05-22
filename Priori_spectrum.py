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
import SolarSpotsAnalysis as ssa 
%reload_ext autoreload
%autoreload 2


def gaussianSpectrum(w):
    ax2 = -np.flip(w)
    ax2 = np.delete(ax2, -1)
    x = np.concatenate((w, ax2))
    return scipy.stats.norm.pdf(x)

def gaussianSpectrum2(w): 
    return scipy.stats.norm.pdf(w)

def back_to_autocorr(w):
    spec = gaussianSpectrum(w)
    gamma = np.real(np.fft.ifft(spec))
    n = len(gamma)
    if n % 2 == 1: 
        gamma = gamma[: (n - 1) // 2]
    else: 
        gamma = gamma[: n // 2]
    T_max = dt * (len(gamma) - 1)
    time = np.linspace(0, T_max, len(gamma))
    gamma *= n / (2 * T_max)  #with 2 * w_n is good accuracy 
    #  gamma *= df / n 
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
    w_n = 5 #Niquyst (Ni) frequency 
    dt = 1 / (2 * w_n) #Niquyst time interval for sampling
    N = 1000 #Total number of points 
    ax = np.linspace(0, w_n, N)
    df = w_n / (N - 1) # = 1 / (2 T_max)
    N2 = 499  #is the lenght of the noise vector created 
    M = int(2 * N2 / np.log(2 * N2))
    dts = dt * 2
    c = back_to_autocorr(ax)
    t = np.linspace(0, (N2-1) * dts , N2) #time range of sampling
    cov = constructCovariance(c, t)
    noise = gauss_noise(cov) 
    meth = 'FPE' #FPE, CAT, OBD, numbers
    P, a_k = ssa.burgMethod(noise, M, meth)
    ssa.spectrum(P, a_k, ax, dts)
    plt.plot(ax, scipy.stats.norm.pdf(ax), color = 'r', alpha = .7)
    title = 'M: {}, met: {}, ARord: {}, N: {}, N2: {}'.format(M, meth, a_k.size - 1, N, N2)
    plt.title(title)
    
    # fft.ifft(function(ax))).shape[0]
    #plt.plot(ax, spectrum(ax))
    #n = len(ax)
    #gamma = np.real(np.fft.ifft(gaussianSpectrum(ax)))[: n // 2]
    #plt.xlim(0,.1)