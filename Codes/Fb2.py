# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:19:42 2020

@author: Workplace
"""
import numpy as np
import sys

def Fastburg(data, m, method = 'FPE', dr = 1):
    N = len(data)
    #Define autocorrelation
    c = np.zeros(m + 2)
    for j in range(m + 1):
        c[j] = data[: N - j] @ data[j : ]
    #Initialize variables
    a = [np.array([1])]
    P = [c[0] / N]
    r = 2 * c[1]
    g = np.array([2 * c[0] - data[0] * data[0].conj() - data[-1] * data[-1].conj(),
                   r])
    #Initialize lists to save arrays
    optimizer = np.zeros(m)
    ks = []
    gs = []
    Drs = []
    rs = []
    #Loop for the FastBurg Algorithm
    for i in range(m):
        sys.stdout.write('\r%f Fast Burg: ' %(i / (m)))
        #Update prediction error filter and reflection error coefficient
        k, new_a = updateCoefficients(a[-1], g)
        #Update variables. Check paper for indeces at j-th loop. 
        r = updateR(data, i, r, c[i + 2])
        #Construct the array in two different, equivalent ways. 
        if dr == 1: DrA = np.dot(constructDr(data, i, new_a, Drs), new_a) 
        if dr == 2: DrA = constructDr2(data, i, new_a, Drs)
        #Update last coefficient
        g = updateG(g, k, r, new_a, DrA)
        #Append values to respective lists
        a.append(new_a)
        P.append(P[-1] * (1 - k * k.conj()))
        ks.append(k)
        gs.append(g)
        rs.append(r)
        #Compute optimizer value for chosen method
        optimizer[i] = optimizeM(P, a[-1], len(data), i + 1, method)
    return P, a, ks, gs, Drs, rs
    
def updateCoefficients(a, g):
    a = np.concatenate((a, np.zeros(1)))
    k = - (np.dot(a.conj(), g[::-1])) / (np.dot(a, g))
    aUpd = a + k * a[::-1].conj()
    return k, aUpd

def updateR(data, i, r, aCorr):
    N = len(data)
    rUp = np.array([2 * aCorr])
    rDown = r - data[: i + 1] * data[i + 1].conj() - \
        data[N - i - 1 : ].conj()[::-1] * data[N - i - 2]
    return np.concatenate((rUp, rDown))
    
    
def constructDr(data, i, a, Drs):
    #print(i, 'Outer')
    N = len(data)
    data1 = data[ : i + 2][::-1]
    data2 = data[N - i - 2 :]
    d1 = -np.outer(data1, data1.conj())
    d2 = -np.outer(data2.conj(), data2)
    Drs.append(d1 + d2)
    return d1 + d2 

def constructDr2(data, i, a, Drs): 
    N = len(data)
    data1 = data[ : i + 2][::-1]
    data2 = data[N - i - 2 :]
    d1 = - data1 * (data1 @ a).conj() 
    d2 = - data2.conj() * (data2 @ a.conj())
    Drs.append(d1 + d2)
    return - data1 * (data1 @ a).conj() - data2.conj() * (data2 @ a.conj()) 
    
    
def updateG(g, k, r, a, Dra):
    gUp = g + (k * g[::-1]).conj() + Dra
    gDown = np.array([r @ a.conj()])
    return np.concatenate((gUp, gDown))

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
    