# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:52:43 2020

@author: Workplace
"""
import numpy as np 
import sys
from MESAAlgorithm import MESA


class Parameter(object): 
    
    def __init__(self, prior, _min, _max, update_step = 1):
        priors = ['uniform', 'jeffreys']
        if prior.lower() in priors: 
            self.prior = prior.lower() 
        else: 
            raise ValueError('Choice for prior not available. Prior must be {}'\
                             .format(' or '.join(priors)))
        self.min = _min
        self.max = _max 
        self.value = np.random.uniform(self.min, self.max)
        self.step = update_step
        self.proposal = self.update()
        
    def setValue(self): 
        if self.prior == 'uniform':
            return np.random.uniform(self.min, self.max)
        elif self.prior == 'jeffreys':
            return np.random.power(2 ) * (self.max - self.min) + self.min 
        
        
    def update(self, step = None, distribution = 'uniform'):
        distributions = ['uniform', 'normal']
        if step == None: step = self.step
        #Raise Error is distribution is not available 
        if distribution.lower() not in distributions: 
            raise ValueError('Distribution has to be {}'\
                             .format(' or '.join(distributions)))
                
        #Update with Uniform distribution    
        if distribution.lower() == 'uniform':
            self.proposal = self.value + np.random.uniform(-step, step)
        
        #Update with Normal distribution 
        elif distribution.lower() == 'normal': 
            self.proposal = self.value + np.random.normal(0, step)
            
        return self.proposal
    
    def priorValue(self, value = 'current'):
        """Compute the value for the prior for passed value. If value is 
        'current' prior value is computed of current value for parameter. 
        Else if value is 'proposal' prior is computed on the proposal value 
        for the parameter. If value is a int or float value for prior is
        computed at the given value for prior"""
        
        #Check input for value 
        availableVals = ['current', 'proposal']
        
        #If value is string
        if type(value) == str:
            if value not in availableVals: raise ValueError('value must be {}\
                                                            or a number'\
                             .format(', '.join(availableVals))) 
            if value == 'current': v = self.value 
            elif value == 'proposal': v = self.proposal
        
        else:
            if len(value) != 1: raise ValueError('length for value must be one')
            v = value
        
        #Assign value for uniform prior
        if self.prior == 'uniform': 
            if self.min < v < self.max:
                return 0
            else: 
                return - np.inf
            
        #Assign value for jeffreys prior 
        if self.prior == 'jeffreys': 
            if self.min < v < self.max:
                return np.log(1 / v)
            else: 
                return -np.inf
       
    
    def _accept(self): 
        #Accept new value for parameter 
        self.value = self.proposal
        return None
    
    
class Prior(object): 
    
    def __init__(self, *parameters):
        self.parameters = parameters
        self.value = self.evaluate(value = 'current')
        
    
    def evaluate(self, value = 'current'): 
        """Compute Value for the prior on selected element. If value is 'current',
        prior is evaluated on current value for the parameter. If value is
        'proposed', prior value is computed on the proposed value for parameters.
        Value can also be a numpy array whose length is equivalent to the number
        of parameters
        """ 
        #Check value is legitimate input
        availableVals  = ['current', 'proposal']
        if value not in availableVals and type(value) != np.ndarray: 
            raise ValueError('value must be {} or numpy array'\
                             .format(' or '.join(availableVals)))
        if type(value) == np.ndarray and value.size != len(self.parameters):
            raise ValueError('Length of values array must be equivalent to the\
                             number of parameters')
    
        #Compute value for prior
        prior = 0
        for parameter in self.parameters:
            prior += parameter.priorValue(value = value)
            
        if value == 'current': self.value = prior
        return prior

class Data(object):
    
    def __init__(self, x, y): 
        self.x = x 
        self.y = y
        
    def residuals(self, model, *parameters):
        model = Model(self, model)
        return self.y - model(*parameters)
            
        
class Likelihood(object):
    
    def __init__(self, data, df, model, *parameters):
        self.data = data
        self.model = model
        self.parameters = parameters
        self.df = df

    def __call__(self, covariance_matrix, res = None, fft = True, value = 'current'):
        parameters = self.parametersValue(value)
        residuals = self.data.residuals(self.model, *parameters)
        if fft == True:
            fResiduals = np.fft.rfft(residuals)
            exponent = np.abs(fResiduals ** 2) / covariance_matrix
            #Compute value for log likelihood
            logL = - (self.df * (exponent)  \
                    +  len(self.data.x) * np.log(covariance_matrix))  #Is normalization fine? 
            logL = logL.sum()  - 0.5 * logL[0]

            return logL
        
        elif fft == False: 
            return 0 
            
        
        else:
            raise ValueError('fft must be True or False')
    
    def parametersValue(self, value = 'current'): 
        values = []
        if value == 'current':
            for p in self.parameters:
                values.append(p.value)
        elif value == 'proposal': 
            for p in self.parameters: 
                values.append(p.proposal)
        return tuple(values)
            
            
        
        
class Posterior(object):
    
    def __init__(self, likelihood, prior):
        self.likelihood = likelihood 
        self.prior = prior
        self.data = self.likelihood.data
    
    def evaluate(self, covariance_matrix, value = 'current', fft = True):
        lik_value = self.likelihood(covariance_matrix, fft = fft, 
                                          value = value)
        prior_value = self.prior.evaluate(value = value)
        return lik_value + prior_value
    
    def parametersValue(self, value = 'current'): 
        return self.likelihood.parametersValue(value = value)
    

    
    

class Metropolis(object): 
    
    def __init__(self, posterior, N = 1000):
        self.N = N
        self.posterior = posterior

    def realTimeEvaluate(self, dt, f, method = 'FPE', update = 'normal'):
        samples = []
        spectra = []
        samples.append(self.posterior.parametersValue())
        cSpectrum = self.constructSpectrum(dt, f, value = 'current')
        spectra.append(cSpectrum)
        cPosterior = self.posterior.evaluate(cSpectrum, value = 'current')
        acc = 0 #Compute the percentage of accepted samples 
        for _ in range(self.N):
            #Update Parameters 
            self._updateParameters(distribution = update)
            #Compute spectrum on current and proposed value for parameters
            pSpectrum = self.constructSpectrum(dt, f, value = 'proposal')
            pPosterior = self.posterior.evaluate(pSpectrum, value = 'proposal')
            sys.stdout.write('\r%2f Execution percentage' %( 100 * (_ + 1) / self.N))
            #print('Cposterior: {}\nPPosterior: {}'.format(cPosterior, pPosterior))
            #Acceptance condition
            if pPosterior - cPosterior > np.log(np.random.uniform(0,1)): 
                acc += 1
                self._acceptParameters()
                cSpectrum = pSpectrum
                cPosterior = pPosterior
            #Record Value for parameters
            spectra.append(cSpectrum)
            samples.append(list(self.posterior.parametersValue()))
        print('\nAcceptance: {}%'.format(100 * acc / self.N))
        return np.array(samples), np.array(spectra)
            
    def constructSpectrum(self, dt, f, value = 'current', method = 'FPE'):
        """Compute the power spectrum on residuals given dataset on residuals"""
        parameters = self.posterior.parametersValue(value = value)
        model = self.posterior.likelihood.model
        residuals = self.posterior.data.residuals(model, *parameters)
        M = MESA(residuals)
        M.solve()
        return M.spectrum(dt, f)
    
    def _updateParameters(self, distribution): 
        values = []
        for p in self.posterior.prior.parameters:
            p.update(distribution = distribution)
            values.append(p.proposal)
        return tuple(values)
        
    def _acceptParameters(self): 
        """Accept proposal value for parameters defined in prior""" 
        values = []
        for p in self.posterior.prior.parameters:
            p._accept()
            values.append(p.value)
        return tuple(values)
            
        



class Model(object):
    
    def __init__(self, data, model):
        """Generate Models from a given set a data, following the chosen distribution.
        For information about order for the parameters check the documentation for 
        every available model. Model is not case sensitive"""
        #Initialize variables
        availModels = ["exponential", "powerlaw", "line", "cauchy", "sin"]
        self.data = data
        
        #Models should be in models
        if model.lower() in availModels: 
            self.model = model
        else: 
            raise ValueError("'{}' Model not available. Available models are {}"\
                             .format(model, ', '.join(availModels)))
        
    def __call__(self, *parameters):
        
        if self.model.lower() == 'exponential':
            return self.exponential(parameters[0], parameters[1], parameters[2])
        if self.model.lower() =='powerlaw':
            return self.powerLaw(parameters[0], parameters[1], parameters[2])
        if self.model.lower() == 'line':
            return self.line(parameters[0], parameters[1])
        if self.model.lower() == 'cauchy':
            return self.cauchy(parameters[0], parameters[1], parameters[2], 
                               parameters[3])
        if self.model.lower() == 'sin': 
            return self.sin(parameters[0], parameters[1], parameters[2])
        
    #Basic models 
    def expoential(self, base, height, floor): 
        """Generate the expected value if datas were exponentially distributed.
        """
        return height * base ** self.data.x + floor
     
    def powerLaw(self, exponent, height, floor):
        """Generate the expected values for power Law distibuted datas"""
        
        return height * (self.data.x ** exponent) + floor
    
    def line(self, slope, intercept):
        """Genereate expected values for linear prediction""" 
        return slope * self.data.x + intercept
    
    def cauchy(self, mode, scale, height, floor):
        """Return the image for cauchy - distributed datas height parameters
        is to be interpreted as height / scale """
        return height  / (1 + ((self.data.x - mode) / scale)) ** 2 + floor
    
    def sin(self, amplitude, frequency, phase):
        return amplitude * np.sin(frequency * self.data.x + phase)
        
        
        