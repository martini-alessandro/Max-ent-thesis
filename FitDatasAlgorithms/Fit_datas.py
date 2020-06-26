# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:48:00 2020

@author: Workplace
"""
import numpy as np 
import pandas as pd 
import scipy.stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
from two_d_fast_eval import FindHeightForLevel
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class data(object): 
    
    def __init__(self, x, y):
        self.x = x 
        self.y = y 

def models(data, parameters, model= 'Pol'): 
    if model == 'Cauchy': 
        mode, height, scale = parameters
        den = np.pi * scale *  (1 + ((data.x - mode) / scale) ** 2)
        return height / den
    elif model == 'Pol': 
        height, exp = parameters
       # return height * (data.x ** (- exp)) + c
        return np.array([height * (d ** (- exp))  for d in data.x])
    else:
        raise ValueError('{} not an available model'.format(model))

def residual(data, parameters): 
    m = models(data, parameters)
    return np.array([d - m[j,:,:] for j, d in enumerate(data.y)])
    #return np.array([data.y - m[:,i] for i in range(m.shape[1])])


def logUniformPrior(parameters):
    """ 
    Compute the value for the uniform prior for a given value of the 
    parameters. If no parameters are passed, automatically returns 0, 
    so it will not contribute to the posterior 
    """
    height, exp, c = parameters
    if e_min < exp < e_max and c_min <= c <= c_max :
        return 0 
    else: 
        return -np.inf 

def logJeffreysPrior(parameters): 
    """ 
    Compute the value for the Jeffreys prior for a given value of the 
    parameters. If no parameters are passed, automatically returns 0, 
    so it will not contribute to the posterior 
    """
    ret = np.where(parameters > 0, np.log(1 / parameters), - np.inf )
    return ret


def logLikelihood(data, parameters, sigma2 = 1):
   # m = models(data, uni_parameters, jef_parameters, model)
    return - (residual(data, parameters) ** 2).sum(axis = 0) / (2 * sigma2)

def target(parameters, data, sigma2 = 1):
    return -logLikelihood(data, parameters, sigma2)

def logPosterior(data, parameters):
    """
    Calculates the logarithm of the posterior using logL and logPriors. 
    to obtain the wanted results one has to check the order of the given
    parameters, to have the right model. 
    """
    #parameters = np.concatenate((uni_parameters, jef_parameters))
    return logLikelihood(data, parameters) + logJeffreysPrior(parameters[0]) +\
        logJeffreysPrior(parameters[1])
            


def metropolis(data, n_of_points): 
    """Apply metropolis algorithm to sample from the posterior"""
    #m, h, s = np.random.uniform([m_min, h_min, s_min], [m_max, h_max, s_max])
    h, e, c = np.random.uniform([h_min, e_min, c_min], [h_max, e_max, c_max])
    samples = [(h, e, c)]
    posterior = [] 
    N = 0
    acc = 0
    # print('m upd: ', sigma_m, 's upd: ', sigma_s, 'h upd: ', sigma_h)
    # print('h interval: ', h_min, h_max, 's invetval: ', s_min, s_max)
    for i in range(n_of_points):
        print(i)
        new_h = h + np.random.uniform(-sigma_h, sigma_h)
        new_e = e + np.random.uniform(-sigma_e, sigma_e)
        r = logPosterior(data, (new_h, new_e)) -\
            logPosterior(data, (h, e))
        N += 1
        if r > np.log(np.random.uniform(0,1)): 
            h, e = new_h, new_e
            acc += 1
        samples.append((h, e, c))
        posterior.append(logPosterior(data, (h, e, c)))
    print('acceptance percentage', acc/N)
    return np.array(samples), np.array(posterior)

def autocorrelation(x):
    N = len(x)
    X=np.fft.fft(x-x.mean())
    # We take the real part just to convert the complex output of fft to a real numpy float. The imaginary part if already 0 when coming out of the fft.
    R = np.real(np.fft.ifft(X*X.conj()))
    # Divide by an additional factor of 1/N since we are taking two fft and one ifft without unitary normalization, see: https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft
    return R/N

def target2(parameters, data):
    return -logPosterior(data, parameters)

    

if __name__ ==  '__main__': 
    h_min, h_max = 0, 10
    e_min, e_max = .8, 1.3
    ex = np.linspace(e_min, e_max, 1000)
    hx = np.linspace(h_min, h_max, 1000)
    E, H = np.meshgrid(ex, hx)
    T = np.array([1000, 2500, 5000, 10000, 20000, 30000, 50000, 70000])
    Tcat = np.array([1000, 2500, 5000, 10000, 20000, 30000, 50000])
    FPEres = np.array([.00247, .00113, 6.21e-4, 3.53e-4, 2.06e-4, 1.48e-4, 7.13e-5, 5.11e-5])
    OBDres = np.array([.002, 9.01e-4, 4.73e-4, 2.81e-4, 1.32e-4, 9.87e-5, 8.40e-5, 5.6e-5 ])
    CATres = np.array([.0391, .0343, .0301, .0282, .0269, .0267, .0234 ])
    dFPE = data(T, FPEres)
    dOBD = data(T, OBDres)
    dCAT = data(Tcat, CATres)
    Z = logLikelihood(dCAT, (H, E)) + logJeffreysPrior(H) #+ logJeffreysPrior(E)
    #LogP = - logPosterior(dFPE, (H[:,None], E[None,:]))
    fig = plt.figure()
    # ax = fig.add_subplot() 
    ax = fig.gca(projection='3d')
    # Z2 = np.exp(Z)
    # x = [0.85008345]
    # y = [0.94075208]
    # z = [Z.max()]
    
    ax.view_init(20, -10)
    # ax.scatter(x, y, z, marker = 'D', c = 'k', alpha = 1, label = 'Mode')
    surf = ax.plot_surface(E, H, Z, cmap = cm.BrBG_r,
                        linewidth=0, antialiased = True)
    fig.colorbar(surf, shrink= .5, aspect = 4)
    plt.xlabel('Exponent')
    plt.ylabel('Height')
    plt.title('LogPosterior')
    plt.legend()
    plt.show()
    
    #levels = np.sort(FindHeightForLevel(Z2, [.50]))
    # #xFPE: array([0.94075208, 0.85008345]) FPE max
    # #xOBD: array([0.98952021, 0.87613358])
    # #xCAT: array([0.08971748, 0.12290522])
    #vec = minimize(target2, (1, .8), args = (dFPE,))
    # plt.plot(0.08971748, 0.12290522, '.', color = 'r', label = 'Max')
    # plt.xlabel('Height')
    # plt.ylabel('Exponent')
    # plt.title('CAT 80% Credibility region')
    # plt.legend()
    # ax.contour(H,E, Z2, levels)
    
    

# if __name__ == '__main__': 
# #       datas = pd.read_csv('spectrum2.csv').iloc[6000:] 
# #     # datas['spectrum'] /=  datas['spectrum'].max()
# #       datas.columns = ['frequency', 'spectrum'] 
# #       d = data(datas['frequency'], datas['spectrum'])
# #       sigma2 = 1  #variance for error distribution
# #       catSpectrum = pd.read_csv('CatSpectrum')
# #       dc = data(catSpectrum['f'][6000::], catSpectrum['spectrum'][6000::])
# #       #initialize parameters bounds for uniform priors
#         h_min, h_max, sigma_h = 0, 10, .5
#         e_min, e_max, sigma_e = 0, 10, .1
#         c_min, c_max, sigma_c = -2, 2, .1
#         
    #resFPE = minimize(target, (1, .8), args = (dFPE,))
#         resOBD = minimize(target, (1, .8, 0), args = (dOBD,))
#         resCAT = minimize(target, (1, 1, 0), args = (dCAT,))
            
    
    
    
    
    
   
                         