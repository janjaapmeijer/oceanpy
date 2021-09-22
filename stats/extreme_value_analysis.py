# http://mathworld.wolfram.com/GumbelDistribution.html
# https://en.wikipedia.org/wiki/Gumbel_distribution
# http://stackoverflow.com/questions/23217484/how-to-find-parameters-of-gumbels-distribution-using-scipy-optimize

import numpy as np
from scipy import stats
import scipy.optimize as so

#modified pdf (Probability Density Function) and cdf (Cumulative Distribution Function)
# mu = location parameter, beta = scale parameter
def gumbel_pdf(x, mu, beta):
    """Returns the value of Gumbel's pdf with parameters mu and beta at x .
    """
    return (1./beta) * (np.exp(-(((x - mu) / beta) + (np.exp(-((x - mu) / beta))))))

def gumbel_cdf(x, mu, beta):
    """Returns the value of Gumbel's cdf with parameters mu and beta at x.
    """
    return np.exp(-np.exp(-(x-mu)/beta))

def gumbel_invcdf(p, mu, beta):
    """Returns the value of Gumbel's inverse cdf with parameters mu and beta,
    variate p is drawn from the uniform distribution on the interval (0,1).
    """
    return mu - (beta * np.log(-np.log(p)))

def gumbel_params(var):
    mu, beta = so.fmin(lambda p, x: (-np.log(gumbel_pdf(x, p[0], p[1]))).sum(), [0.5,0.5], args=(np.array(var),)) #check 0.5 0.5
    return mu, beta

def gev_params(var):
    c, mu, beta = stats.genextreme.fit(np.array(var))
    return c, mu, beta
# stats.genextreme.cdf()
# plt.plot((1 / (1-stats.genextreme.cdf(xmodel, *gev_params(annualmax)))), xmodel, 'r--', label='GEV fit')
# plt.plot((1 / (1-stats.genextreme.cdf(xmodel, *stats.genextreme.fit(np.array(annualmax))))), xmodel, 'r--', label='GEV fit')

def weibull_params(var):
    a, c, mu, beta = stats.exponweib.fit(np.array(var))
    return a, c, mu, beta
# stats.exponweib.cdf()
# plt.plot((1 / (1-stats.exponweib.cdf(xmodel, *stats.exponweib.fit(np.array(annualmax))))), xmodel, 'm--', label='Weibull fit')

def gpd_params(var):
    ''' Optimisation parameters for the Generalised Pareto Distribution '''
    c, mu, beta = stats.genpareto.fit(np.array(var))
    return c, mu, beta


#Gringorten (1963) plotting position formula
def gringorten_ppf(time,x):
    """ """
    comb = tuple(zip(time,x))
    N = len(x)
    r = sorted(comb, key=lambda x: x[1], reverse=True)

    T_Rr = [(N + 0.12) / (i+1 - 0.44) for i in range(0,N)]
    temp = [r[i] + (T_Rr[i],) for i in range(0,N)]
    sortontime = sorted(temp, key=lambda x: x[0])

    time,x,T_R=zip(*sortontime)
    time = list(time)
    x = list(x)
    T_R = list(T_R)

    y = [-np.log( -np.log(1 - (1 / T_R[i]))) for i in range(0,N)]

    return T_R, y, time, x

def weibull_ppf(time, x):
    comb = tuple(zip(time,x))
    N = len(x)
    r = sorted(comb, key=lambda x: x[1], reverse=True)


    T_Rr = [N / (i + 1) for i in range(0,N)]
    temp = [r[i] + (T_Rr[i],) for i in range(0,N)]
    sortontime = sorted(temp, key=lambda x: x[0])

    time,x,T_R=zip(*sortontime)
    time = list(time)
    x = list(x)
    T_R = list(T_R)

    y = [-np.log( -np.log(1 - (1 / T_R[i]))) for i in range(0,N)]

    return T_R, y, time, x

def confidence_interval(data, confidence=0.95):
    import numpy as np
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t.ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
