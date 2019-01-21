"""
Created on Mon Jan 21 08:37 2018
@author: Rodrigo J. Bombardi
"""
import numpy as np
from scipy.stats import norm

def mk_test_sens_slope(tmp, alpha=0.05):
    """
    Copyright (c) 2017 Michael Schramm

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    This function calculates the Mann-Kendall Test and the Sen's slope calculation. It was
    partially inspired by the function created by Michael Schramm @mps9506
    https://github.com/mps9506/Mann-Kendall-Trend/blob/master/mk_test.py.

    Input:
        tmp:   a vector of data
        alpha: significance level (0.05 default)
    Output:
        slope: the Sen's slope value
        h: True (if trend is present) or False (if trend is absence)
        pval: p value of the significance test
        zmk: normalized test statistics
    Examples
    --------
      >>> tmp = np.random.rand(100)
      >>> slope,h,pval,zmk = mk_test_sens_slope(tmp,0.05)
    """
    num = len(tmp)
    index=np.arange(0,num,1)
    #    Creating arrays with dimension size equal to the number of all possible differences
    sgn = np.zeros((int(num*(num-1)/2)))
    slo = np.zeros((int(num*(num-1)/2)))
    #    Calculaing the indicator function (sgn) and the Sen's slope (slo) using only
    # one loop.
    beg=0                       # first position of sgn vector for the kth iteration
    for kt in range(1,num):     # loop in time
        ned=beg+num-kt          # last position of sgn vector for the kth iteration
        sgn[beg:ned]=tmp[kt:num]-tmp[kt-1]
        slo[beg:ned]=(tmp[kt:num]-tmp[kt-1])/(index[kt:num]-index[kt-1])
        beg=ned
    #    Calculaing the mean
    es = np.sum(np.sign(sgn))
    # ---- Calculating the variance -----
    # calculating the correction term for tied observations (qt = number of data points in each tie group)
    corr=0.
    un = np.unique(tmp)
    for kt in un:
        id=np.where(tmp == kt)
        corr += len(id[0])*(len(id[0])-1.)*(2*len(id[0])+5)
    var = (num*(num - 1.)*(2.*num+5.) - corr)/18.
    if es > 0.:
       zmk = (es-1.)/np.sqrt(var)
    if es == 0.:
       zmk = 0.
    if es < 0.:
       zmk = (es+1.)/np.sqrt(var)
    # calculate the p_value
    pval = 2*(1-norm.cdf(abs(zmk)))  # two tail test
    h = abs(zmk) > norm.ppf(1-alpha/2)
    slope = np.median(slo)
    return slope, h, pval, zmk
