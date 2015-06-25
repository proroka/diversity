# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:56:52 2015
imp
@author: amandaprorok
"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as pp
import numpy.linalg as la
import scipy.cluster.hierarchy as ch
import funcdef_diversity as fd

x = 10 # number of species
t = 2 # number of traits / dimensions

a = np.random.rand(t,x)
d = fd.distance_v(a)

Z = ch.linkage(d,method='single',metric='euclidean')
dend = ch.dendrogram(Z)

b = fd.branch_lengths(Z)
h = fd.branch_presence(Z)
fd = fd.fd(h,b)

print fd

        
         
         
         
         
         
         
         