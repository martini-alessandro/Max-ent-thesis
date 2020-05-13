# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:48:00 2020

@author: Workplace
"""
import numpy as np 
import pandas as pd 
import scipy.stats
import matplotlib.pyplot as plt

class data(object): 
    
    def __init__(self, x, y):
        self.x = x 
        self.y = y 

def models(data, parameters): 
        mode, height, scale = parameters
        den = np.pi * scale *  (1 + ((data.x - mode) / scale) ** 2)
        return height / den

def residual(data, parameters): 
    return data.y - models(data, parameters)

def logUniformPrior(parameters):
    """ 
    Compute the value for the uniform prior for a given value of the 
    parameters. If no parameters are passed, automatically returns 0, 
    so it will not contribute to the posterior 
    """
    m, h, s = parameters
    if m_min < m < m_max and h_min < h < h_max :
        return 0 
    else: 
        return -np.inf 

def logJeffreysPrior(parameters): 
    """ 
    Compute the value for the Jeffreys prior for a given value of the 
    parameters. If no parameters are passed, automatically returns 0, 
    so it will not contribute to the posterior 
    """
    m, h, s = parameters
    if s_min < s < s_max:
        return np.log(1 / s)
    else: 
        return -np.inf


def logLikelihood(data, parameters):
   # m = models(data, uni_parameters, jef_parameters, model)
    return - (residual(data, parameters) ** 2).sum() / (2 * sigma2)

def logPosterior(data, parameters):
    """
    Calculates the logarithm of the posterior using logL and logPriors. 
    to obtain the wanted results one has to check the order of the given
    parameters, to have the right model. 
    """
    #parameters = np.concatenate((uni_parameters, jef_parameters))
    return logLikelihood(data, parameters) + logUniformPrior(parameters) +\
            logJeffreysPrior(parameters)


def metropolis(data, n_of_points): 
    """Apply metropolis algorithm to sample from the posterior"""
    #m, h, s = np.random.uniform([m_min, h_min, s_min], [m_max, h_max, s_max])
    m, h, s = np.random.uniform([0.0079, h_min, s_min], [.0079, h_max, s_max])
    samples = [(m, h, s)]
    posterior = [] 
    N = 0
    acc = 0
    print('m upd: ', sigma_m, 's upd: ', simga_s, 'h upd: ', sigma_h)
    print('h interval: ', h_min, h_max, 's invetval: ', s_min, s_max)
    for i in range(n_of_points):
        new_m = m #+ np.random.normal(0, sigma_m)
        new_h = h + np.random.uniform(0, sigma_h)
        new_s = s + np.random.uniform(0, sigma_s)
        r = logPosterior(data, (new_m, new_h, new_s)) -\
            logPosterior(data, (m, h, s))
        N += 1
        if r > np.log(np.random.uniform(0,1)): 
            m, h, s = new_m, new_h, new_s
            acc += 1
        samples.append((m, h, s))
        posterior.append(logPosterior(data, (m, h, s)))
    print(' acceptance percentage', acc/N)
    return np.array(samples), np.array(posterior)


def autocorrelation(x):
    """
    Compute the autocorrelation of the chain
    """
    N = len(x)
    X=np.fft.fft(x-x.mean())
    # We take the real part just to convert the complex output of fft to a real numpy float. The imaginary part if already 0 when coming out of the fft.
    R = np.real(np.fft.ifft(X*X.conj()))
    # Divide by an additional factor of 1/N since we are taking two fft and one ifft without unitary normalization, see: https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft
    return R/N

if __name__ == '__main__': 
    datas = pd.read_csv('spectrum2.csv').iloc[6000:] 
   # datas['spectrum'] /=  datas['spectrum'].max()
    datas.columns = ['frequency', 'spectrum'] 
    d = data(datas['frequency'], datas['spectrum'])
    sigma2 = 1  #variance for error distribution
    catSpectrum = pd.read_csv('CatSpectrum')
    dc = data(catSpectrum['f'][6000::], catSpectrum['spectrum'][6000::])
    #initialize parameters bounds for uniform priors
    m_min = 0.007
    m_max = .009
    m_up = 1e-5
    h_min = .1
    h_max = 1e4
    h_up = .1
    s_min = .00001
    s_max = .001
    s_up = 5e-9
    sigma_m = 1e-5 
    sigma_h = 1e-2
    sigma_s = 1e-2
    #uMinVec = np.array([0, 1])
    #uMaxVec = np.array([1, 1e3])
    #uMinVec = np.array([0, 0])  #min values for x[0]=mode,
    #uMaxVec = np.array([.1 , 10e6]) #max values for x[0]=mode, x[1]=height,
    #initialize parameters bounds for jeffreys priors 
    #jMinVec = np.array([0]) #bouns for scale factor
    #jMaxVec = np.array([.01])
    
    
    
    
    
   
                         
        