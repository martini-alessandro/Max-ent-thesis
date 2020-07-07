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
import sys
# %reload_ext autoreload
# %autoreload 2

class data:
    def __init__(self, x, y): 
        self.x = x 
        self.y = y
        
def autocorrelation(x, norm = 'N'):
    N = len(x)
    X=np.fft.fft(x-x.mean())
    # We take the real part just to convert the complex output of fft to a real numpy float. The imaginary part if already 0 when coming out of the fft.
    R = np.real(np.fft.ifft(X*X.conj()))
    # Divide by an additional factor of 1/N since we are taking two fft and one ifft without unitary normalization, see: https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft
    if norm == 'N':
        return R/N
    elif norm == None: 
        return R
    else:
        raise ValueError('this normalization is not available')
    
def updatePredictionCoefficient(x, reflectionCoefficient):
    new_x = np.concatenate((x, np.zeros(1)))
    flip_x = reflectionCoefficient * np.flip(new_x)
    return new_x + flip_x
        
def optimizeM(P, a_k, N, m, method): #removed noise
    if method == 'FPE':
        #print('P: {} \n'.format(P[-1]))
        return P[-1] * (N + m + 1) / (N - m - 1)
    elif method == 'CAT': 
        P = np.array(P[1:])
        k = np.linspace(1, m, m)
        PW_k = N / (N - k) * P
        #print('P: {} \n value {}\n'.format(P[-1], PW_k[-1]))
        return 1 / (N * PW_k.sum())- (1 / PW_k[-1])
    elif method == 'OBD': 
        P_m = P[-1]
        P = np.array(P[:-1])
        return (N - m - 2)*np.log(P_m) + m*np.log(N) + np.log(P).sum() + (a_k**2).sum()
    elif isinstance(method, int):
        pass
    
    # elif method == 'logL':
    #     spec = spectrum(P[-1], a_k, freq, dt, False)['spectrum']
    #     return .5 * np.sum(noise / spec).real
    else: 
        raise ValueError('{} is not a an available method'.format(method))        
    
def burgMethod(data, m, method = 'FPE'):
    #initialization of variables
    print('Standard!')
    P_0 = (data ** 2).mean() 
    P = [P_0]
    a_0 = 1 
    a_k = [np.array([a_0])]
    _f = np.array(data) 
    _b = np.array(data)
    optimizer = np.zeros(m) 
    k_list = []
    #Burg's recursion 
    for i in range(m):
        sys.stdout.write('\r%f Normal Burg: ' %(i / (m - 1)))
        f = _f[1:]
        b = _b[:-1]
        den = np.dot(f, f) + np.dot(b, b)
        k = - 2 * np.dot(f.T, b) / den
        k_list.append(k)
        a_k.append(updatePredictionCoefficient(a_k[i], k))
        P.append(P[i] * (1 - k * k.conj()))
        _f = f + k * b
        _b = b + k * f 
        #print('P: ', P, '\nak: ', a_k[-1])
        optimizer[i] = optimizeM(P, a_k[-1], len(data), i + 1, method)
        
    #selecting the minimum for the optimizer and recording its position 
    if method == 'CAT' or method == 'FPE' or method == 'OBD':  
    #CAT optimizer[0] is 0, so we need the minimum value of optimizer[1]
        op_index = optimizer[1:].argmin() 
    #+ 1 because of slicing [1:], +1 because opt[i] correspond to P[i+1], ak[i+1]
        op_index += 2
    elif isinstance(method, int):
        #raise error if method < M 
        op_index = method
    else: 
        raise ValueError('method selected is not allowable')
    return P, np.array(a_k), k_list



        
def spectrum(P, a_k, f, dt = 1, plot = True): 
    N = a_k.shape[0]
    den = sum([a_k[k] * np.exp(2 * np.pi * 1j * k * f * dt) for k in range(N)])
    spec = dt * P / (np.abs(den) ** 2)
    df = pd.DataFrame({'f': f, 'spectrum': spec}, index = None)
    if plot == True: 
        plt.plot(f, spec)
        # plt.xlabel('frequency {}'.format('1/months'))
        # plt.ylabel('spectrum')
    return df


# if __name__ == '__main__': 
#     datas = pd.read_csv('zuerich-monthly-sunspot-numbers-.tsv', sep = '\t',
#                         index_col = 0, 
#                         header = None, 
#                         names = ['Spots'],
#                         parse_dates = True, 
#                         infer_datetime_format = True )[:1000] 
#     datas.index = datas.index.to_period('m')
#     datas -= datas.mean() 
#     dt = 1 
#     noise = np.fft.rfft(datas['Spots']) * dt 
#     noise2 = noise * noise.conj()
#     freq = np.fft.rfftfreq(datas.shape[0], d = dt)
    
#     n = datas.shape[0]
#     M = int(2 * n / np.log(2))

#             # plt.plot(noise2)
#     P1, a_k1 = burgMethod(datas['Spots'], 200, 'logL')
            
    
        
            
    # plt.xlim(0,100)
    # f = np.linspace(0,.01, 1000)
    # spec1 = spectrum(P1, a_k1, f, plot = False)
    # plt.plot(spec1['f'][500:], spec1['spectrum'][500:])
    # _datas = datas[: n // 2]
    # P, a_k = burgMethod(_datas['Spots'], 200, 'CAT')
    # spec = spectrum(P, a_k, f, plot = False)
    # plt.plot(spec['f'][500:], spec['spectrum'][500:], label = 'half data')
    # #mu = datas.mean() 
    # #f = np.linspace(0,.01, 10000)
 
    
    