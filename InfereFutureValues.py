# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:38:34 2020

@author: Alessandro Martini 
"""

import pandas as pd 
import numpy as np 
import SolarSpotsAnalysis as ssa
import matplotlib.pyplot as plt

def infereFuture(P, ak, datas, number_of_datas): 
    ak = np.delete(ak, 0)
    weights = - ak
    data = np.flip(datas)
    points = np.array([])
    for i in range(number_of_datas):
        data = data[:len(ak)]
       # print(ak[0], data[0], '\n', ak[1], data[1], '\n', ak[2], data[2])
        new_point = weights.T @ data
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
    P, a_k = ssa.burgMethod(new_datas, 200, 'CAT')
    fut = infereFuture(P, a_k, new_datas, 800)
    f = np.linspace(0,.01, 1000) 
    P1 , a_k1 = ssa.burgMethod(fut, 80, 'CAT')
    spec = ssa.spectrum(P1, a_k1, f, plot = False)['spectrum']
    plt.plot(f, spec)
    plt.xlim(.008, .009)
    # fut = infereFuture(P, a_k, new_datas, 2000)
    # error = np.sqrt(P)
    # plt.plot(np.array(datas['Spots'][2001:2102]),'.', color = 'orange')
    # plt.plot(fut, color = 'r')
    # x = np.linspace(0, 100, 100)
    # plt.fill_between(x, fut - error, fut + error)