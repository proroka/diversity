# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 09:37:47 2015
@author: amanda

"""



import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.image as img
import networkx as nx
import sys
import time
import pickle






# -----------------------------------------------------------------------------#
def generate_Q(S, U):
    # fill all rows and columns with at least one 1
    if S==U:
        Q = np.identity(S) 
    else:
        Q = np.zeros((S,U))
        if S<U:
            np.fill_diagonal(np.transpose(Q),1,wrap=True)
            padc = range(S,U,S+1)
            padr = np.random.randint(0,S,len(padc))        
            Q[padr, padc] = 1    
        else:
            np.fill_diagonal(Q,1,wrap=True)        
            padr = range(U,S,U+1)
            padc = np.random.randint(0,U,len(padr))        
            Q[padr, padc] = 1        

 
    num = np.random.randint(0,U*S+1)    
    num = 0
    # fill matrix    
    for n in range(num):
        i = np.random.randint(0,S)
        j = np.random.randint(0,U)
        Q[i,j] = 1
        
    rk = np.linalg.matrix_rank(Q)    
       
    
    print Q
    #print 'Q with S =', S, 'has rank ', rk , 'with ', num, 'extra 1s'
    print 'S = ', S
    print 'R = ', rk    
    plt.imshow(Q, interpolation='none')
    plt.show()
    
    return Q, rk
    
# -----------------------------------------------------------------------------#

num_species = 5
num_traits = 10

generate_Q(num_species, num_traits)

