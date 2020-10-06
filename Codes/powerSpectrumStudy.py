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
from scipy.signal import welch, tukey

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
        i += int(length * (1 - overlap))
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
    return np.array(spectra).mean(0)
    

def computeSpectrum(noise, method, dt, freq): 
    if type(noise) == list:
        spectra = []
        optimizers = []
        orders = [] 
        for i in range(len(noise)):
            sys.stdout.write('\r%f Method: {}'.format(method) \
                             %((i + 1) / len(noise)))
            M = MESA(noise[i])
            P, ak, opt = M.solve(optimisation_method = method)
            spectrum = M.spectrum(dt, freq)
            spectra.append(spectrum)
            optimizers.append(opt)
            orders.append(ak.size)
        return np.array(spectra), np.array(optimizers), np.array(orders)
   
        


if __name__ == '__main__': 
    dt, T = 1 / 10, 300
    f = np.linspace(0, 5, 1000)
    psd = scipy.stats.norm.pdf(f, 2.5, .5)
    noises = [] 
    Simulations = 1000
    for i in range(Simulations):
        np.random.seed(i)
        sys.stdout.write('\r%f Noise generation' %((i + 1) / Simulations))
        t, noise, Gaussfreq, fnoise, ps = noiseGen.generate_data(f, psd, T, 
                                                      sampling_rate = 1 / dt)
        noises.append(noise)
    
    CATspectrum, CATopt, CATorders = computeSpectrum(noises, 'CAT', dt, freq)
    FPEspectrum, FPEopt, FPEorders = computeSpectrum(noises, 'FPE', dt, freq)
    OBDspectrum, OBDopt, OBDorders = computeSpectrum(noises, 'OBD', dt, freq)
    

    #Sampling 
    dt, T, = 1 / 2048, 8
    f, psd = np.loadtxt('..{}LIGO-P1200087-v18-AdV_DESIGN_psd.dat'.format(os.sep), unpack = True)
    LigoNoises = []
    #Generate noise
    for i in range(500): 
        np.random.seed(i)
        sys.stdout.write('\r%2f' %((i + 1)/500))
        t, noise, freq, fnoise, ps = noiseGen.generate_data(f, psd, T, 
                                                          sampling_rate = 1 / dt)
        LigoNoises.append(noise)
        
    LigoCATspectrum, LigoCATopt, LigoCATorders = computeSpectrum(LigoNoises, 'CAT', dt, freq)
    LigoFPEspectrum, LigoFPEopt, LigoFPEorders = computeSpectrum(LigoNoises, 'FPE', dt, freq)
    LigoOBDspectrum, LigoOBDopt, LigoOBDorders = computeSpectrum(LigoNoises, 'OBD', dt, freq)
    
        
        
    # N = 2048 
    # sampling_rate = 1 / dt
    # overlap_fraction = .5
    # segment_duration = N * dt #lunghezza sottoarray
    # padding = 0.4 / segment_duration #frazione della parte 'crescente descrescente' 
    # window_function  = tukey(int(sampling_rate*segment_duration), padding)
    # wf, ws = welch(noise, fs  = 1 / dt,
    #                          window = window_function,
    #                          nperseg         = N,
    #                          noverlap        = int(overlap_fraction*N),
    #                          return_onesided = False,
    #                          scaling         = 'density') #dopo N // 2 frequ negative
  