# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:45:14 2020

@author: Workplace
"""
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from scipy.optimize import minimize 

class data(object):
    def __init__(self, x, y): 
        self.x = x 
        self.y = y

def logLikelihood(data, parameters, sigma2 = 1, model = 'Pol'):
    r = residuals(data, parameters, mod = model)
    return - (r ** 2).sum() / 2 * sigma2 

def residuals(data, parameters, mod = 'Pol'): 
    return data.y - model(data, parameters, mod)

def model(data, parameters, model = 'Pol'): 
    h, e = parameters 
    ret = h * (data.x ** - e)
    return ret
    
def logJefPrior(parameter):
    jef = np.where(parameter > 0, np.log(1 / parameter), - np.inf)
    return jef
    
def logUniformPrior(parameter): 
    return 0 

def logPosterior(data, parameters, model = 'Pol'):
    h, e = parameters
    return logLikelihood(data, (h, e)) #+ logJefPrior(h)# + logJefPrior(e)

def target(parameters, data): 
    return - logPosterior(data, parameters, model = 'Pol')

if __name__ == '__main__': 
    #defining the prior domains 
    h_min, h_max = 0, 1.5
    e_min, e_max = .85, 1.3
    ex = np.linspace(e_min, e_max, 1000)
    hx = np.linspace(h_min, h_max, 1000)
    E, H = np.meshgrid(ex, hx)
    T = np.array([1e3, 2.5e3, 5e3, 10e3, 20e3, 30e3, 50e3, 70e3])
    FPEres = np.array([.00247, .00113, 6.21e-4, 3.53e-4, 2.06e-4, 1.48e-4, 7.13e-5, 5.11e-5])
    OBDres = np.array([.002, 9.01e-4, 4.73e-4, 2.81e-4, 1.32e-4, 9.87e-5, 8.40e-5, 5.6e-5 ])
    CATres = np.array([.0391, .0343, .0301, .0282, .0269, .0267, .0234 ])
    d = data(T, FPEres)
    dOBD = data(T, OBDres)
    dCAT = data(T[: CATres.size], CATres)
    
    
    