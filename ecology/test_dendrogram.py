# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:56:52 2015
imp
@author: amandaprorok
"""

import numpy as NP
import pylab as PL
import matplotlib.pyplot as PP
import matplotlib.image as IM
import numpy.linalg as LA
import scipy.cluster.hierarchy as CH
import funcdef_diversity as FD

x = 10 # number of species
t = 2 # number of traits / dimensions

a = NP.random.rand(t,x)
d = FD.distance_v(a)

Z = CH.linkage(d,method='single',metric='euclidean')
dend = CH.dendrogram(Z)

b = FD.branch_lengths(Z)
h = FD.branch_presence(Z)
fd = FD.fd(h,b)

print fd

        
         
         
         
         
         
         
         