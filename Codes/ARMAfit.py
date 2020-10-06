# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:48:46 2020

@author: Workplace

This code is meant to test different criterions to choose the order of an
autoregressive process, testing them all on different sets of knowkn ARMA(p, 0)
processes. 

"""

import numpy as np
from MESAAlgorithm import MESA
import statsmodels.tsa.arima_process as arima
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import os
from numpy.polynomial import Polynomial

def _generateCoefficients(order, min_ = -.99, max_ = .99):

    coefficients = np.random.uniform(min_, max_, order)
    
    print('size of coeff: ', coefficients.size)
    p = Polynomial([1])
    for c in coefficients:
        p *= Polynomial([1, c])
    print('size of pol: ', p.coef.size)
    return p.coef[p.coef > 1e-10] 

def _generateComplexCoeff(order, _min = 0, _max = .99):
    mod = np.random.uniform(_min, _max, order)
    phase = np.random.uniform(0, 2 * np.pi, order)
    coefficients = mod * np.exp(1j * phase)
    print('size of coeff: ', coefficients.size)
    p = Polynomial([1])
    for c in coefficients:
        p *= Polynomial([1, c])
    print('size of pol: ', p.coef.size)
    return p.coef
    
def simulateProcess(length = 1000, order = 10, scale = 1, min_ = 0, max_ = .99): 
    coeff = _generateCoefficients(order, min_, max_)
    MAcoeff = np.ones(1)
    return lfilter(MAcoeff, coeff, np.random.normal(0, scale, length)), coeff


if __name__ == '__main__': 
    length = 50000
    order = 10
    scale = 3e-8
    _min, _max = -.9, .9
    dt = 1
    freq = np.linspace(0, 2, 1000)
    data, coeff = simulateProcess(length, order, scale, _min, _max)
    # plt.plot(np.log10(np.abs(coeff)))
    M = MESA(data)
    N = MESA(data)
    P, ak, opt = M.solve()
    N.solve()
    N.a_k = coeff
    N.P = scale ** 2
    # plt.plot(freq, M.spectrum(dt, freq))
    # plt.plot(freq, N.spectrum(dt, freq))
    plt.plot(coeff, '.')
    
