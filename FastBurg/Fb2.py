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
    c = np.zeros(m + 5)
    for j in range(m + 5):
        c[j] = data[: N - j] @ data[j : ]
    #Initialize variables
    a = [np.array([1])]
    P = [c[0] / N]
    r = 2 * c[1]
    g = np.array([2 * c[0] - data[0] * data[0].conj() - data[-1] * data[-1].conj(),
                   r])
    #List to save arrays
    optimizer = np.zeros(m)
    ks = []
    gs = []
    Drs = []
    rs = []
    #Loop for the FastBurg Algorithm
    for i in range(m):
        #sys.stdout.write('\r%f Fast Burg: ' %(i / (m - 1)))
        #Update prediction error filter and reflection error coefficient
        k, new_a = updateCoefficients(a[-1], g)
        ks.append(k)
        #Save prediction error filter and P values
        a.append(new_a)
        P.append(P[-1] * (1 - k * k.conj()))
        #Update variables 
        r = updateR(data, i, r, c[i + 2])
        rs.append(r)
        #Construct the array in two different, equivalent ways and compare them
        if dr == 1: DrA = np.dot(constructDr(data, i, new_a, Drs), new_a) 
        if dr == 2: DrA = constructDr2(data, i, new_a, Drs)
        #check if functions give same result
        #Update last coefficient
        g = updateG(g, k, r, new_a, DrA)
        gs.append(g)
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
    