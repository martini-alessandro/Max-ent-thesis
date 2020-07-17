# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:05:31 2020

@author: Workplace

Generate noise of a given power spectrum
"""

import numpy as np 
import scipy.stats
import os
import matplotlib.pyplot as plt
import FastBurg2 as fb
from MESAAlgorithm import MESA
from scipy.interpolate import interp1d

def psd(f):
    N = len(f)
    #return np.concatenate((np.repeat(5e20, N // 2),np.repeat(3, N // 2)))
    return scipy.stats.norm.pdf(f, 3, .5) 

def generate_noise(f, psd, dt, t_min = 0, f_min = None, N = 10000):
    
    #Generate the minimum value of frequency
    if f_min == None: f_min = f[0]
    
    #Interpolate frequency and power spectral density
    interpPSD = interp1d(f, psd)
    
    #Compute time and frequency variables
    Ny = 1 / (2 * dt)   #Nyquist frequency
    T =  N * dt         #Total time
    df = 1 / T          #Frequency interval
    time = np.linspace(t_min, t_min + T, 2 * (N -1)) 
    frequencies = np.linspace(f_min, Ny, N)

   
    #Generate Frequency-Domain noise
    FDnoise = np.sqrt(interpPSD(frequencies) * df / 2) * np.exp(1j * np.random.uniform(0, 2 * np.pi, N)) * np.random.normal(size = N) #Multiplication for random gaussian is needed to have limit of psd as variance, not as upper - lower limit
    
    #convert noise in time domain 
    TDnoise = np.fft.irfft(FDnoise) * 2 * (N -1) #2(N-1) is the number of point of FFT
    return time, TDnoise, frequencies, FDnoise, interpPSD(frequencies)

def generate_noise_roughly(f, psd):
    #Define variables
    N = len(psd)
    Ny = f[-1]
    dt = 1 / (2 * Ny)
    T = dt * N
    df = 1 / T
    
    #Generate Noise
    FDnoise = np.sqrt(psd * df / 2) * np.exp(1j * np.random.uniform(0, 2 * np.pi, N)) \
    #* np.random.normal(0, 1, N)
    TDnoise = np.fft.irfft(FDnoise) * (2 * N)  
    return FDnoise, TDnoise

if __name__ == '__main__':
    # f = np.linspace(0, 6, n)
    # ps = psd(f) 
    f, ps = np.loadtxt('..{}LIGO-P1200087-v18-AdV_DESIGN_psd.dat'.format(os.sep), unpack = True)
    dt = 1 / (2 * f[-1])
    time, noise, freq, fnoise, PSD = generate_noise(f, ps, dt, N = f.size)
    #fNoise, noise = generate_noise_roughly(f, ps)
    
    M = MESA(noise)
    P, ak = M.solve(optimisation_method = 'FPE', method = 'Fast')
    f1 = np.linspace(0, f[-1], len(f))
    plt.loglog(f, M.spectrum(dt, f1))
    plt.loglog(f, ps, color = 'r', linestyle = '--')