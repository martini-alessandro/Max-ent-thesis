# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:13:49 2020

@author: Workplace
"""

import numpy as np
import MetropolisHasting 
import MESAAlgorithm 

class Metropolis(object): 
    
    def __init__(self, data, likelihood_distribution, model, *parameters):
        self.data = data
        self.posterior = Posterior(self.data, likelihood_distribution, model)
        self.samples = []
        
    def __call__(self, N, *parameters):
        if self.realtime = True: 
            realTimeSolve(... )
        elif self.realtime = False: 
            solve(...)
        else:
            raise ValueError('real time has to be True or False')
    
    
    def realTimeSolve(self, N, *parameters):
        for _ i range(N): 
            sys.stdout.write('\r%f' %((_ + 1) / N))
            #Estimate residuals and compute PSD
             
            self.
            
        
    def solve(self, N, *parameters): 
        for _ in range(N): 
            sys.stdout.write('\r%f' % ((_ + 1) / N))
            #Updata parameters value to compute they proposal value
            self._updateParameters(*parameters)
            
            #Compute posterior difference
            r = self.posterior.proposal(*parameters) -\
                self.posterior.value(*parameters)
            
            #accept values if condition met
            if r > np.log(np.random.uniform(0, 1)): 
        
                self._acceptNewValues(*parameters)
    
            #update samples array
            self._updateSamples(*parameters)
        
        return np.array(self.samples)
            
    def _updateParameters(self, *parameters):
        #If parameter object, update its value, otherwise pass
        for p in parameters: 
            if type(p) == Parameter:
                p._update()
        return None
    
    def _acceptNewValues(self, *parameters):
        #Accept proposed value for parameters object 
        for p inparameters:
            if type(p) == Parameter:
                p._accept()
        return None
    
    def _updateSamples(self, *parameters):
        newSamples = []
        for p in parameters:
            if type(p) == Parameter: newSamples.append(p.value)
            else: newSamples.append(p)
        self.samples.append(newSamples)
        return None
    
    
    
    
if __name__ == '__main__'
    