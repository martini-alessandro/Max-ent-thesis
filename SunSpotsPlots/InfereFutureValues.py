# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:38:34 2020

@author: Alessandro Martini 
"""

import pandas as pd 
import numpy as np 
import SolarSpotsAnalysis as ssa
import matplotlib.pyplot as plt
from Fit_datas import autocorrelation
%autoreload 2

def infereFuture(P, ak, datas, number_of_datas): 
    """
    Uses the optimal filter results to compute the estime for the new value
    of the time series using a N'th order autoregressive process, whose order
    corresponds with the ak size 
    """
    #ak = np.delete(ak, 0)
    #weights = - ak
    weights = - np.delete(ak, 0)
    data = np.flip(datas)
    points = np.array([])
    for i in range(number_of_datas):
        data = data[:len(weights)]
        new_point = weights.T @ data + np.random.normal(0, np.sqrt(P)) #second parameter is the std! 
        while new_point < 0:
            new_point = weights.T @ data + np.random.normal(0, np.sqrt(P))
        data = np.insert(data, 0, new_point)
        points = np.append(points, new_point)
        # datas = np.append(datas, new_point)
    return points 

if __name__ == '__main__': 
    datas = pd.read_csv('zuerich-monthly-sunspot-numbers-.tsv', sep = '\t',
                        index_col = 0, 
                        header = None, 
                        names = ['Spots'],
                        parse_dates = True, 
                        infer_datetime_format = True ) 
    # datas['Spots'] -= datas['Spots'].mean() 
    datas.index = datas.index.to_period('m')
    new_datas = np.array(datas['Spots'][:2001])
    M = int(2 * new_datas.shape[0] / np.log(2 * new_datas.shape[0]))
    P, a_k = ssa.burgMethod(new_datas, M)
   
    plt.plot(np.array(datas['Spots'][2001:]), color = 'b', 
              label = 'Observed', linestyle = 'dotted')
    l = []
    for i in range(100):
        print(i)
        fut = infereFuture(P, a_k, new_datas, 800)
        l.append(fut)
        #plt.plot(fut, lw = 0.5, alpha = .6)
    l = np.array(l)
    m, l5, l95 = np.percentile(l, [50, 5, 95], axis = 0)
    ax = np.linspace(0, 800, 800)
    plt.fill_between(ax, l5, l95, alpha = .5)
    plt.plot(m, color = 'k')
    plt.title('Max Ent estimate for future solar spots numbers')
    plt.xlabel('Time [Months]')
    plt.ylabel('# Spots')
    plt.xlim(0, 800)
    
    
    
 