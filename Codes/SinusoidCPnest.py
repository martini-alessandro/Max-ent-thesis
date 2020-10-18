import unittest
import numpy as np
import cpnest.model
from scipy import stats
from MESAAlgorithm import MESA
import os
from scipy.signal import welch, tukey

class Data(object):
    
    def __init__(self, x, y): 
        self.x = x 
        self.y = y
        
    def residuals(self, model, *parameters):
        model = Model(self, model)
        return self.y - model(*parameters)
    
    
class SinusoidModel(cpnest.model.Model):
    """
    An n-dimensional gaussian
    """
    def __init__(self, data, psd, df, model = 'sin2'): 
        self.names = ['Amplitude1', 'Amplitude2', 'Frequency']
        self.bounds = [[5, 20], [5, 20], [5,10]]
        self.data = data
        self.psd =  psd 
        self.model = model
        self.df = df
        self.exp = []
    
    def log_likelihood(self, x):
        A1, A2, F = x['Amplitude1'], x['Amplitude2'], x['Frequency']
        residuals = self.data.residuals(self.model, A1, A2, F)
        k = len(self.data.x)
        Fresiduals =  np.fft.rfft(residuals)
        exp = (np.abs(Fresiduals) ** 2)
        self.exp.append(exp)
        logL = - (exp / k  + k * np.log(self.psd))
        logL = logL.sum() - logL[0]
        return logL
        
    def log_prior(self, p):
        logP = super(SinusoidModel,self).log_prior(p)
#        for n in p.names: logP += -0.5*(p[n])**2
        return logP
    

#     def __init__(self, data, dt, freq, model = 'sin'):
#         self.names= ['Amplitude', 'Frequency', 'Phase']
#         self.bounds=[[23, 24], [0, 10], [0, 2 * np.pi]]
# #        self.bounds[0] = [-50, 50]
#         self.data = data
#         self.dt = dt
#         self.freq = freq
#         self.df = freq[1] - freq[0]
#         #Models should be in models
#         self.model = model

    # def log_likelihood(self, x):
    #     A, F, P = x['Amplitude'], x['Frequency'], x['Phase']
        
    #     #Compute Power Spectral Density 
    #     residuals = self.data.residuals(self.model, A, F, P)
    #     M = MESA(residuals)
    #     M.solve() 
    #     k = len(self.data.x)
    #     covariance_matrix = M.spectrum(self.dt, self.freq)
    #     fResiduals = np.fft.rfft(residuals)
        
    #     exponent = np.abs(fResiduals ** 2) / covariance_matrix
    #     #Compute value for log likelihood
    #     logL = - (exponent / k +  k * np.log(covariance_matrix))  #Is normalization fine? 
    #     logL = logL.sum()  - logL[0]
 
    #     return logL
        
#     def log_prior(self, p):
#         logP = super(SinusoidModel,self).log_prior(p)
# #        for n in p.names: logP += -0.5*(p[n])**2
#         return logP
    

class Model(object):
    
    def __init__(self, data, model):
        """Generate Models from a given set a data, following the chosen distribution.
        For information about order for the parameters check the documentation for 
        every available model. Model is not case sensitive"""
        #Initialize variables
        availModels = ["exponential", "powerlaw", "line", "cauchy", "sin", "sin2",\
                       "gauss"]
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
        if self.model.lower() == 'sin2':
            return self.sin2(parameters[0], parameters[1], parameters[2])
        if self.model.lower() == 'gauss':
            return self.gauss(parameters[0], parameters[1], parameters[2])
        
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
    
    def sin2(self, amplitude1, amplitude2, frequency): 
        return amplitude1 * np.sin(frequency * self.data.x) +\
               amplitude2 * np.cos(frequency * self.data.x)
               
    def gauss(self, amplitude, mean, size): 
        return stats.norm.pdf(self.data.x, mean, size) * amplitude 
        
def Welch(data, N, dt, overlap_fraction = .5): 
    sampling_rate = 1 / dt
    segment_duration = N * dt #lunghezza sottoarray
    padding = 0.4 / segment_duration #frazione della parte 'crescente descrescente' 
    window_function  = tukey(int(sampling_rate*segment_duration), padding)
    wf, ws = welch(data, fs  = 1 / dt, window = window_function, nperseg = N,
                              noverlap        = int(overlap_fraction*N),
                              return_onesided = False,
                              scaling         = 'density') #dopo N // 2 frequ negative
    return wf, ws 

if __name__== '__main__':
    amplitude1, amplitude2, frequency= 8, 14, 7.3
    #Generate Data 
    NumberOfData = 500
    dt = 1 / (2 * frequency)
    x = np.arange(0, NumberOfData ) * dt
    T = NumberOfData  * dt
    error = np.random.normal(0, .03 * amplitude1, NumberOfData)
    # y = amplitude * np.sin(frequency * x + phase)
    y = amplitude1 * np.sin(frequency * x) +\
        amplitude2 * np.cos(frequency * x) +\
        error
    D = Data(x, y)
    
    # Define Frequency arrays 
    Ny = 1 / (2 * frequency)
    freq = np.arange(0, NumberOfData / 2 + 1) / T 
    
    M = MESA(error)
    M.solve() 
    spectrum = M.spectrum(dt, freq)
    # spectrum = np.repeat(error.var(), len(freq))
    # frequency, spectrum = Welch(D.y, NumberOfData /2 + 1, dt)
    
    # Initialize NS 
    c = SinusoidModel(D, spectrum, 1 / T, 'sin2')
    work=cpnest.CPNest(c,   verbose      =  2,
                            poolsize     = 32,
                            nthreads     = 2,
                            nlive        = 1024,
                            maxmcmc      = 1000,
                            output       = os.getcwd())
    
    work.run()
    print('log Evidence {0}'.format(work.NS.logZ))
    x = work.posterior_samples.ravel()
    np.savetxt('PosteriorSamples.txt', x)
    np.savetxt('exp.txt', c.exp)
    

 
