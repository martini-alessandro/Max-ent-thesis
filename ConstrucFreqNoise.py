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
    return scipy.stats.norm.pdf(x, 3, 0.5)
    
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
    times   = np.linspace(starttime, starttime + T , N) #should be start_time + T - dt
    # filter out the bad bits
    kmin = np.int(fmin/df)
    kmax = np.int(fmax/df)+1
    # generate the FD noise
    frequencies      = df*np.arange(0, N / 2 + 1) #df * N / 2 is Ny frequency, + 1 needed because a_range cuts last term
    frequency_series = np.zeros(len(frequencies), dtype = np.complex128)
    print('kmin: {}, kmax: {}, freqmax: {}'.format(kmin, kmax, frequencies[-1]))
    if zero_noise is False:
        for i in range(kmin, kmax):
            print("percentage: {}".format((i - kmin + 1) / (kmax - kmin)))
            sigma = np.sqrt(psd_int(frequencies[i]) / df) * np.sqrt(.5) # why this? 
            frequency_series[i] = np.random.normal(0, sigma) + 1j * np.random.normal(0, sigma)
    # inverse FFT to return the TD strain
    time_series = np.fft.irfft(frequency_series, n = N) * df * N 
    return times, time_series, frequencies, frequency_series, psd_int(frequencies), 
if __name__ == '__main__': 
    dt = 1 / 10 #if dt is too small (like 1 / 100), result is awful
    T = 1000
    Ny = 1 / (2 * dt)
    f_max = Ny #I choose my max frequency to be the Nyquist frequency! Smallest is 0
    N = T / dt 
    print(N)
    #to avoid interpolation errors, f must include Ny frequency
    f = np.linspace(0, Ny + 1, 100 * int(Ny)) 
    sd = gaussSpectrum(f)
    times, time_series, freq, freqSeries, psd = generate_data(f, sd, T, 0,
                                                              1 / dt, 
                                                              0, f_max)
    
    #t_noise = time_series[: f_max * T] #the number of point I generated. Not sure of this 
    t_noise = time_series[: time_series.size // 2]
    print(t_noise.size)
    M = int(2 * t_noise.size / np.log(2 * t_noise.size))
    P, a_k = ssa.burgMethod(t_noise, M)
    print('ak size is {}'.format(a_k.size))
    g = np.linspace(0, 5, 3000)
    spec = ssa.spectrum(P, a_k, g, dt)
    plt.plot(f, sd, color = 'r')
    