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
        new_point = weights.T @ data + np.random.normal(0, np.sqrt(P))
        while new_point < 0:
            new_point = weights.T @ data + np.random.normal(0, np.sqrt(P))
        data = np.insert(data, 0, new_point)
        points = np.append(points, new_point)
        datas = np.append(datas, new_point)
    return points

if __name__ == '__main__': 
    datas = pd.read_csv('zuerich-monthly-sunspot-numbers-.tsv', sep = '\t',
                        index_col = 0, 
                        header = None, 
                        names = ['Spots'],
                        parse_dates = True, 
                        infer_datetime_format = True ) 
    datas.index = datas.index.to_period('m')
    new_datas = np.array(datas['Spots'][:2001])
    P, a_k = ssa.burgMethod(new_datas, 200, 8)
    fut = infereFuture(P, a_k, new_datas, 2600)
    # plt.plot(np.array(datas['Spots'][2001:]), color = 'b', 
    #          label = 'Observed', linestyle = 'dotted')
    plt.plot(fut, color = 'orange', label = 'estimated')
    plt.title('Max Ent estimate for future solar spots numbers')
    plt.xlabel('Time [Months]')
    plt.ylabel('# Spots')
    #plt.ylim(top = 220)
    plt.legend(loc = 2)
 