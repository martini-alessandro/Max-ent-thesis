# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:28:56 2020

@author: Alessandro Martini 

Apply MESA to solar spots. Main goal: find a 11 years periodicity
"""
import pandas as pd 
import numpy as np 
import scipy.stats
import scipy
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

class data:
    def __init__(self, x, y): 
        self.x = x 
        self.y = y
    
def updatePredictionCoefficient(x, reflectionCoefficient):
    new_x = np.concatenate((x, np.zeros(1)))
    flip_x = reflectionCoefficient * np.flip(new_x)
    return new_x + flip_x
        
def optimizeM(P, a_k, N, m, method): 
    if method == 'FPE':
        return P[-1] * (N + m + 1) / (N - m - 1)
    elif method == 'CAT': 
        P = np.array(P[1:])
        k = np.linspace(1, m, m)
        PW_k = N / (N - k) * P
        return 1 / (N * PW_k.sum())- (1 / PW_k[-1])
    elif method == 'OBD': 
        P_m = P[-1]
        P = np.array(P[:-1])
        return (N - m - 2)*np.log(P_m) + m*np.log(N) + np.log(P).sum() + (a_k**2).sum()

    else: 
        raise ValueError('{} is not a an available method'.format(method))        
    
def burgMethod(data, m, method = 'FPE'):
    #initialization of variables
    P_0 = (data ** 2).mean() 
    P = [P_0]
    a_0 = 1 
    a_k = [np.array([a_0])]
    _f = np.array(data) 
    _b = np.array(data)
    optimizer = np.zeros(m) 
    #Burg's recursion 
    for i in range(m):
        f = _f[1:]
        b = _b[:-1]
        den = np.dot(f, f) + np.dot(b, b)
        k = - 2 * np.dot(f.T, b) / den
        a_k.append(updatePredictionCoefficient(a_k[i], k))
        P.append(P[i] * (1 - k ** 2))
        _f = f + k * b
        _b = b + k * f 
        optimizer[i] = optimizeM(P, a_k[-1], len(data), i + 1, method)
    #selecting the minimum for the optimizer and recording its position 
    if method == 'CAT': 
        optimizer[0] = optimizer[1] + 1
    op_index = optimizer.argmin() + 1 
    return P[op_index], a_k[op_index]

def spectrum(P, a_k, f, dt = 1, plot = True): 
    N = a_k.shape[0]
    den = sum([a_k[k] * np.exp(2 * np.pi * 1j * k * f * dt) for k in range(N)])
    spec = dt * P / (np.abs(den) ** 2)
    df = pd.DataFrame({'f': f, 'spectrum': spec}, index = None)
    if plot: 
        plt.plot(f, spec)
        plt.xlabel('frequency {}'.format('1/months'))
        plt.ylabel('spectrum')
    return df

"""def spectrum2(P, a_k, f, dt = 1, plot = True):
    #Trying to implement spectrum avoiding python loop. Hidden loops in doing
    #so turn out to be even slower than original spectrum function 
    N = a_k.shape[0]
    k = np.linspace(0, N - 1, N)
    k = k.repeat(len(f))
    k.shape = (N, len(f))
    den = a_k.T @ np.exp(2 * np.pi * 1j * k * f * dt)
    spec = dt * P / (np.abs(den) ** 2)
    df = pd.DataFrame({'f': f, 'spectrum': spec}, index = None)
    if plot: 
        plt.plot(f, spec)
        plt.xlabel('frequency {}'.format('1/months'))
        plt.ylabel('spectrum')
    return df """

if __name__ == '__main__': 
    datas = pd.read_csv('zuerich-monthly-sunspot-numbers-.tsv', sep = '\t',
                        index_col = 0, 
                        header = None, 
                        names = ['Spots'],
                        parse_dates = True, 
                        infer_datetime_format = True ) 
    datas.index = datas.index.to_period('m')
    n = datas.shape[0]
    P1, a_k1 = burgMethod(datas['Spots'], 200, 'CAT')
    f = np.linspace(0,.01, 1000)
    spec1 = spectrum(P1, a_k1, f, plot = False)
    plt.plot(spec1['f'][500:], spec1['spectrum'][500:])
    _datas = datas[: n // 2]
    P, a_k = burgMethod(_datas['Spots'], 200, 'CAT')
    spec = spectrum(P, a_k, f, plot = False)
    plt.plot(spec['f'][500:], spec['spectrum'][500:], label = 'half data')
    #mu = datas.mean() 
    #f = np.linspace(0,.01, 10000)
 
    
    