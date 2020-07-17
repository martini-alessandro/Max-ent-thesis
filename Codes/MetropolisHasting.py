# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:05:38 2020

@author: Alessandro Martini

This class is meant to implement a Markov Chain Monte Carlo algorithm, to 
explore the posterior space parameter and make inference on models via 
the reduction of residuals squared
"""
import numpy as np 

class Data(object): 
    def __init__(self, x, y, e): 
        self.x = x 
        self.y = y 
        self.e = e
        
class Likelihood(): 
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data, *parameters): 
         return None 
     
     
class Parameter(object):
    
    def __init__(self, _min, _max, value = None):
        """Initialize class variables"""
        self.min = _min
        self.max = _max
        #consider adding prior in __init__ . Might be easier to implement in Metropolis, when calling posterior 
        if value == None: self._setValue()
            
    def _setValue(self):
        """Compute random value for the parameter between max and min values"""
        self.value = np.random.uniform(self.min, self.max)
        return None 
    
    def update(self, upd):
        """Update the value of the parameter and record it in self.newValue"""
        self.newValue = self.value + np.random.uniform(-upd, upd)
        return self.newValue
    
    def prior(self, prior, value = None, log = True):
        p = Prior(prior) 
        return p(self, value, log)


class Prior(object):
        def __init__(self, prior):
            priors = ['Uniform', 'Jeffreys']
            self.prior = prior
            if self.prior not in priors:
                raise ValueError("Invalid prior: possible choices are \
                                         {}".format("'Uniform' and 'Jeffreys'"))
        def __call__(self, parameter, value = None, log = True): 
            
            #Set value if no value is passed to be the value of the parameter
            if value == None: value = parameter.value
            
            #Compute prior if value is in parameter's domain 
            if parameter.min < value < parameter.max:
                if self.prior == 'Uniform':
                    if log: return 0
                    else: return 1
                elif self.prior == 'Jeffreys':
                    if log: return np.log(1 / value)
                    else: return 1 / value
                    
            #Compute prior if value is not in parameter's domain
            else: 
                if log: return -np.inf
                else: return 0

            
# class Metropolis(object): 
#     def __init__(self, *args, **kwargs):

# class model(object):
#     def __init__(self, *args, **kwargs):
