# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:12:11 2020

@author: Workplace
"""

import numpy as np
from scipy.optimize import minimize
import scipy.stats
import matplotlib.pyplot as plt
import MetropolisHasting as mcmc

def likelihood(parameters, data):
    exp, height = parameters
    mean = height * data.x ** exp
    logL = - .5 * ((data.y - mean) ** 2).sum()
    if exp < -1 or exp > 0: return -np.inf
    else: return logL

def Try(parameters): 
    x = np.linspace(5000, 100000, 2000)
    h = 1
    mean = h * x ** parameter
    logL = -.5 * (data.y - ...).sum()
    
    
    
x = np.array([3, 5, 10, 20, 30, 50, 70, 100]) * 1000
y = np.array([.874, .7, .551, .42, .377, .312, .294, .255])
d = mcmc.Data(x, y, 0)
plt.plot(x, y, '.')
d = mcmc.Data(x, y, 0)
exp = mcmc.Parameter(-1, 0, 'uniform', .05)
h = mcmc.Parameter(0, 10, 'uniform', .1)
m = mcmc.Metropolis(10000, d, 'norm', 'powerlaw', exp, h, 0)
# samples = m.solve()