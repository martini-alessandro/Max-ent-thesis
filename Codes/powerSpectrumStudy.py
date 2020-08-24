# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:50:05 2020

@author: Workplace
"""

import ConstrucFreqNoise as noiseGen
from MESAAlgorithm import MESA
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sys
import os
from scipy.signal import welch

def gaussSpectrum(f, mean, size):
    return scipy.stats.norm.pdf(f, mean, size)


def percentageError(estimate, real):
    return np.abs(estimate - real) / real


def split(vector, length, overlap): 
    subVector = [] 
    i = 0
    j = 0
    while i < len(vector):
        j += 1
        subVector.append(vector[i : int(i + length)])
        i += int(length * overlap)
    print('Number of noise vectors is {}'.format(j))
    return np.array(subVector)


def alternativeCAT(dt, f, vector, length, overlap = .5):
    subVectors = split(vector, length, overlap)
    spectra = []
    for v in subVectors:
        M = MESA(v)
        P, ak, opt = M.solve(optimisation_method = 'CAT')
        spectrum = M.spectrum(dt, f)
        spectra.append(spectrum)
    return np.array(spectra)
    
def computeSpectrum(N, f, psd, T, dt, optimizer): 
    print('N is: {}\nOptimizer is: {}'.format(N, optimizer))
    spectra = []
    orders = []
    residuals = []
    for i in range(int(N)):
        t, noise, freq, fnoise, ps = noiseGen.generate_data(f, psd, T, 
                                                      sampling_rate = 1 / dt)
        M = MESA(noise)
        P, ak, opt = M.solve(optimisation_method = optimizer)
        orders.append(ak.size)
        spectrum = M.spectrum(dt, freq)
        residuals.append(percentageError(spectrum, ps).mean())
        spectra.append(spectrum)
    spectra = np.array(spectra)
    meanSpectrum = np.array(spectra).mean(axis = 0)
    
    return freq, spectra, ps, orders, residuals

    
if __name__ == '__main__': 
   # np.random.seed(7) #5, 12, are a lucky ones
    
    #Sampling 
    dt, T = 1 / 2048, 20
    #Compute Spectrum
    f, psd = np.loadtxt('..{}LIGO-P1200087-v18-AdV_DESIGN_psd.dat'.format(os.sep), unpack = True)
    #Generate noise
    t, noise, freq, fnoise, ps = noiseGen.generate_data(f, psd, T, 
                                                      sampling_rate = 1 / dt)
    alCAT = alternativeCAT(dt, freq, noise, 5000, .5)
    plt.loglog(freq, alCAT.mean(0))
    plt.loglog(freq, ps)
    plt.xlim(10, 1024)
    ####
    M = MESA(noise)
    P, ak, opt = M.solve(optimisation_method = 'CAT')
    spectrum = M.spectrum(dt, freq)
    plt.loglog(freq, spectrum, '--', color = 'k')
    