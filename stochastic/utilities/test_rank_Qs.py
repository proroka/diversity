# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:02:22 2015

@author: amandaprorok
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

from generate_Q import *

# -----------------------------------------------------------------------------#


num_species = 4
num_traits = 4
num_nodes = 8
num_iter = 1

QQ = np.zeros((num_traits*num_nodes+num_species, num_species*num_nodes))
QQ_rks = np.zeros((num_species,num_iter))
Q_rks = np.zeros((num_species,num_iter))

print "num rows: ", QQ.shape[0]
print "num cols: ", QQ.shape[1]

for it in range(num_iter):
    print it
    for ui in range(1,num_species+1):
        rk = 0    
        sys.stdout.flush()
        while rk!=ui:
            Q, rk, s = generate_Q(num_species, ui)
            #print Q
        # generate QQ
        QQ = np.zeros((ui*num_nodes+num_species, num_species*num_nodes))
        QQ[0:ui*num_nodes,:] = sp.linalg.block_diag(*([Q.T]*num_nodes))
        QQ[ui*num_nodes:,:] = np.concatenate([np.identity(num_species)]*num_nodes,axis=1)
        rk_QQ = np.linalg.matrix_rank(QQ)
    
        ind = ui-1
        Q_rks[ind,it]= rk
        QQ_rks[ind,it] = rk_QQ
        

coef = QQ_rks / Q_rks

ind = 0
a = Q_rks[ind,:].flatten()
b = QQ_rks[ind,:].flatten()
c = Q_rks.flatten()
d = QQ_rks.flatten()
plt.scatter(c,d)
plt.show()


