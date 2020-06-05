# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:02:22 2020

@author: Alessandro Martini
"""

import pandas as pd 
import numpy as np 
import scipy.stats 


def optimizeM(P, a_k, N, m, method): #removed noise
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
    elif isinstance(method, int):
        pass
    
    # elif method == 'logL':
    #     spec = spectrum(P[-1], a_k, freq, dt, False)['spectrum']
    #     return .5 * np.sum(noise / spec).real
    else: 
        raise ValueError('{} is not a an available method'.format(method)) 
        
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
    
def fastBurg(data, m, method = 'FPE'): 
    """Compute the fast Burg Algorithm """ 
    #initialize variables 
    N = len(data)
    c = autocorrelation(data, norm = None)[: m + 3]
    P = [c[0] / N]
    ak = [np.array([1])]
    optimizers = np.zeros(m + 1)
    g = np.array([2 * c[0] - np.abs(data[0]) - np.abs(data[-1]),
                  2 * c[1]])
    r = np.array(2 * c[1])
    for i in range(m + 1):
        print(i / (m + 1))
        k, new_a = updateCoefficients(ak[i], g)
        ak.append(np.array(new_a))  
        P.append(P[i] * (1 - k * k.conj()))
        optimizers[i] = optimizeM(P, ak[-1], N, i + 1, method)
        #we update r. at 0 order we defined r_1 and use it to compute r_2, so we
        #call updateR with index i + 2.  
        r = updateR(data, i + 2, c, r)
        #Update g
        #Constructing g_i + 1, we have to call a_(1 + 1), so new_a.
        #Dr only appears as ascalar
        
        Dra = np.dot(constructDr(data, i), new_a)
        g = updateG(g, k, r, new_a, Dra)
            
    if method == 'CAT' or method == 'FPE' or method == 'OBD':  
        op_index = optimizers[1:].argmin() + 1 
    elif isinstance(method, int):
    #raise error if method < M 
        op_index = method
    else: 
        raise ValueError('method selected is not allowable')
        
    return P[op_index], ak[op_index]
    
def updateCoefficients(a, g):
    """Updates predictio coefficiente and compute reflection coefficient"""
    addZero = np.concatenate((a, np.zeros(1)))
    k = - np.dot(addZero.conj(), np.flip(g)) / np.dot(addZero, g)
    new_a = addZero +  k * np.flip(addZero.conj())
    return k, new_a
   
def updateR(data, i, c, r):
    #index i is calld as "j + 2", so that we are always computing "r_{i} = r_{j + 2}" (j refers to fastBurg loop)
    #constructing r_i we use c[i]
    r_0 = np.array([2 * c[i]])
    
    #compute the 'second' element for r array. Constructing r_i, in the first term
    #datas are going from data[0] data[i - 2] and multiplied by data[i - 1] 
    r_1 = r - data[: i - 1] * data[i - 1].conj()
    
    #Here is last element. Datas array goes from data[N] to data N - (i - 1) and have to be flipped
    r_2 = np.flip(data[len(data) - i + 1 : len(data)].conj()) * data[len(data) - i] 
   
    return np.concatenate((r_0, r_1 - r_2))

def constructDr(data, i):
    #Dr is matrix constructed from data arrays
    data1 = np.flip(data[ : i + 2])
    #from 'last - i, we are computing idex = (i + 2), and data has to be last - index = last - i - 2
    data2 = data[len(data) - i - 2 : len(data)] 
    return - np.outer(data1, data1.conj()) - np.outer(data2.conj(), data2)
    

def updateG(g, k, r, a, Dr):
    g1 = g + k.conj() * np.flip(g.conj()) + Dr
    g2 = np.array([np.dot(r, a.conj())])
    #print(g1, g2)
    return np.concatenate((g1, g2))

if __name__ == '__main__': 
    datas = pd.read_csv('zuerich-monthly-sunspot-numbers-.tsv', sep = '\t',
                        index_col = 0, 
                        header = None, 
                        names = ['Spots'],
                        parse_dates = True, 
                        infer_datetime_format = True )
    datas.index = datas.index.to_period('m')
    datas = np.array(datas['Spots'])
    N = len(datas)
    M = int(2*N / np.log(2*N))