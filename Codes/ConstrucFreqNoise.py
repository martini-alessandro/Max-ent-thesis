# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:50:17 2020

@author: Workplace
"""

import numpy as np
import scipy.stats 
import pandas as pd 
import SolarSpotsAnalysis as ssa 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys

def gaussSpectrum(x): 
    return scipy.stats.norm.pdf(x, 3, 0.5)# + scipy.stats.norm.pdf(x, 1, .3)
    
def generate_data(f,
                  psd,
                  T= 16.0,
                  starttime = 0,
                  sampling_rate = 100000.,
                  fmin = 0,
                  fmax = 100,
                  zero_noise = False,
                  asd = False):
    # f, psd = np.loadtxt(psd_file, unpack=True)
    if asd is True : psd *= psd
    # generate an interpolant for the PSD
    psd_int = interp1d(f, psd, bounds_error=False, fill_value=np.inf)
    df      = 1 / T
    N       = int(sampling_rate * T)
    times   = np.linspace(starttime, starttime + T , N) 
    # filter out the bad bits
    kmin = np.int(fmin/df)
    kmax = np.int(fmax/df) + 1
    # generate the FD noise
    print(N)
    frequencies      = df * np.arange(0, N / 2 + 1) #df * N / 2 is Ny frequency, + 1 needed because arange cuts last term
    frequency_series = np.zeros(len(frequencies), dtype = np.complex128)
    # print('kmin: {}, kmax: {}, freqmax: {}'.format(kmin, kmax, frequencies[-1]))
    if zero_noise is False:
        print(kmin, kmax, frequencies.size)
        sigma = np.sqrt(psd_int(frequencies[kmin: kmax + 1]) / df)
        phi = np.random.uniform(0, 2 * np.pi, len(sigma))
        frequency_series = sigma * np.exp(1j * phi) #* np.random.normal(0, 1, size = len(sigma)) 
        
        
        # for i in range(kmin, kmax):
        #     sys.stdout.write('\r%f perc of noise' %((i - kmin - 1) / (kmax - kmin)))
            
        #     sigma = np.sqrt(psd_int(frequencies[i]) / df) * .5
    
        #     frequency_series[i] = np.random.normal(0, sigma) + 1j * np.random.normal(0, sigma)
    # inverse FFT to return the TD strain
    time_series = np.fft.irfft(frequency_series) * df * (N + 1)
    return times, time_series, frequencies, frequency_series, psd_int(frequencies)

def residual(frequency, burg_spectrum):
    r2 = (gaussSpectrum(frequency) - burg_spectrum) ** 2 #compute residuals squared
    return r2.mean()

# if __name__ == '__main__': 
#     np.random.seed(1) #Fissa l'inizializzazione alla catena pseudo-random
#     dt = 1 / 1024     #if dt is too small (like 1 / 100), result is awful
#     T = 3             #Max time of observation 
#     Ny = 1 / (2 * dt) #Compute Ny Frequency 
#     f_max = Ny        #I choose my max frequency to be the Nyquist frequency! Smallest is 0
#     N = T / dt        #Total number of points 
#     #Generate the frequency spectrum, to avoid interpolation errors, f must include Ny frequency
#     #f = np.linspace(0, Ny + 1, 10000) 
#     #sd = gaussSpectrum(f)
#     residuals = np.array([])
#     g = np.linspace(0, 5, 1000)
#     spectra = []
#     rng = 1
#     asize = []
#     #f, psd = np.loadtxt('LIGO-P1200087-v18-AdV_DESIGN_psd.dat', unpack = True)
#     for _ in range(rng):
        
#         print(_)
#         times, time_series, freq, freqSeries, psd1 = generate_data(f, psd, T, 0,
#                                                               1 / dt, 
#                                                               10, f_max)
#         M = int((2 * N) / (np.log(2 * N)))
#         p, ak, k = ssa.burgMethod(time_series, M, M)
#         ak = np.array(ak)
#         p1, ak1 = fba.fastBurg(time_series, M, M)
#         sp = ssa.spectrum(p, ak, freq, dt, plot = False)['spectrum']
#         # plt.loglog(freq, sp)
#         # plt.loglog(f, psd, linestyle = '--')
#         # plt.xlim(10, 512)
#         # plt.ylim(1e-47, 1e-40)
#         #r = residual(g, sp)
#         #residuals = np.append(residuals, r)
#         #asize.append(ak)
        
    
    
    
    
    
    # for _ in range(rng):
    #     print('{} \r'.format(_ / rng))
    #     times, time_series, freq, freqSeries, psd = generate_data(f, sd, T, 0,
    #                                                           1 / dt, 
    #                                                           0, f_max)
    #     t_noise = time_series[: time_series.size // 2] #we only want "positive frequency" noise
    #     M = int(2 * t_noise.size / np.log(2 * t_noise.size)) #Compute the maximum order for the AR process 
    #     P, a_k, optimizer = ssa.burgMethod(t_noise, M, 'CAT')
    #     #P, a_k = fba.fastBurg(t_noise, M)
    #     spec = ssa.spectrum(P, a_k, g, dt, plot = False)
    #     spectra.append(spec['spectrum'])
    #     r2, r2p = residual(g, spec['spectrum'])
    #     residuals = np.append(residuals, r2)
    #     asize.append(a_k.size)
    # #plt.plot(g, spec['spectrum'], label = 'estimate')
    # spectra = np.array(spectra)
    # s = spectra.mean(0)
    # p5, p95 = np.percentile(spectra, (5, 95), axis = 0)
    # plt.plot(g, s, label = 'mean', color = 'k')
    # plt.plot(g, gaussSpectrum(g), color = 'r', label = 'original', 
    #           linestyle = '--')
    # plt.fill_between(g, p5, p95, alpha = .5, label = '90% Cr')
    # plt.xlabel('frequency [a.u.]')
    # plt.ylabel('Power spectral density [a.u.]')
    # plt.legend()
   
        
    