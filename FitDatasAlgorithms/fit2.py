# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:37:33 2020

@author: Alessandro Martini

Class that allows to fit datas using metropolis algorithm, in a most general
way
"""
import numpy as np 
import scipy.stats 

class data(object):
    
    def __init__(self, x, y, error = 0):
        self.xvalue = x
        self.yvalue = y
        self.error = error

class parameter(object):
    
    def __init__(self, _min, _max, upd, prior = 'Uniform'): 
        self.min = _min 
        self.max = _max 
        self.upd = upd
        self.value = self.random()
        self.prior = prior

    def random(self):
       return np.random.uniform(self.min, self.max)
   
    def update(self): 
        return self.value + np.random.uniform(-self.upd, self.upd)

    def prior(self, value = self.value):  
        
        if self.prior == 'Uniform': 
            if self.min < value < self.max:
                return 0
            else: 
                return -np.inf
            
        elif self.prior == 'Jeffreys':
            if self.min < value < self.max :
                return np.log(1 / self.value())
            else:
                return -np.inf


def model(datas, parameters, model = 'Pol'):
    if model == 'Pol': 
        h, e = parameters
        return h * datas.x ** e

def residuals(datas, parameters, model): 
    return data.y - model(datas, parameters, model = model)

def logLikelihood(datas, parameters, model, var = 1): 
    """Compute value of logLikelihoodFunction for a Gaussian likelihood""" 
    r2 = residuals(datas, parameters, model) ** 2
    return r2.sum() / (2 * var)

def logPosterior(datas, parameters, model, var = 1):
    e, h = parameters
    return logLikelihood(datas, parameters, model, var) + 

def metropolisHasting(N, datas, parameters, model):
    N = 0
    acc = 0
    exp, h = parameters
    samples = (exp.value, h.value)
    for i in range(N):
        exp_proposal, h_proposal = exp.update(), h.update()
        r = exp.prior()
    
if __name__ == '__main__': 
    exp = parameter(-10, 0, 1, 'Uniform')
    h = parameter(0, 1, .1, 'Uniform')
    metropolisHasting(100, datas, (exp, h), 'Pol')
    

