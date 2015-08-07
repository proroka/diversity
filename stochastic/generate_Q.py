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
    
    verbose = 0
    
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
    #num = 0
    # fill matrix    
    for n in range(num):
        i = np.random.randint(0,S)
        j = np.random.randint(0,U)
        Q[i,j] = 1
        
    rk = np.linalg.matrix_rank(Q)    
    s = int(np.sum(np.sum(Q)))
    
    if verbose:
        print Q
        print 'S = ', S
        print 'U = ', U
        print 'R = ', rk    
        print '1 = ', s
        plt.imshow(Q, interpolation='none')
        plt.show()
        
    
    return Q, rk, s
    
# -----------------------------------------------------------------------------#

num = 5

# -----------------------------------------------------------------------------#

num_species = 10
num_traits = 10
q_rs = np.zeros((num_species, num_traits, num))
q_rs1 = np.zeros((num_species, num_traits, num))
q_rs2 = np.zeros((num_species, num_traits, num))
run = 'S10_U10'

max_rk = np.min([num_species, num_traits])


not_full = True
nrs = 0
nrs1 = 0
nrs2 = 0
cnt_rs = 0
cnt_rs1 = 0
cnt_rs2 = 0
while not_full:
    Q, rk, s = generate_Q(num_species, num_traits)
    if rk==max_rk:
        q_rs[:,:,nrs] = Q
        print 'Found rk=S, num = ', nrs
        if nrs<num-1:
            nrs += 1
        cnt_rs += 1            
    if rk==(max_rk-1):
        q_rs1[:,:,nrs1] = Q
        print 'Found rk=S-1, num = ', nrs1
        if nrs1<num-1:        
            nrs1 +=1
        cnt_rs1 += 1    
    if rk==(max_rk-2):
        q_rs2[:,:,nrs2] = Q
        print 'Found rk=S-2, num = ', nrs2
        if nrs2<num-1:        
            nrs2 +=1
        cnt_rs2 += 1    
    if (cnt_rs>=num and cnt_rs1>=num and cnt_rs2>=num):
        not_full = False


prefix = "./data/Q/" + run + "_"

pickle.dump(q_rs, open(prefix+"q_rs.p", "wb"))
pickle.dump(q_rs1, open(prefix+"q_rs1.p", "wb"))
pickle.dump(q_rs2, open(prefix+"q_rs2.p", "wb"))

# -----------------------------------------------------------------------------#

num_species = 10
num_traits = 15
q_rs = np.zeros((num_species, num_traits, num))
q_rs1 = np.zeros((num_species, num_traits, num))
q_rs2 = np.zeros((num_species, num_traits, num))
run = 'S10_U15'

max_rk = np.min([num_species, num_traits])

not_full = True
nrs = 0
nrs1 = 0
nrs2 = 0
cnt_rs = 0
cnt_rs1 = 0
cnt_rs2 = 0
while not_full:
    Q, rk, s = generate_Q(num_species, num_traits)
    if rk==max_rk:
        q_rs[:,:,nrs] = Q
        print 'Found rk=S, num = ', nrs
        if nrs<num-1:
            nrs += 1
        cnt_rs += 1            
    if rk==(max_rk-1):
        q_rs1[:,:,nrs1] = Q
        print 'Found rk=S-1, num = ', nrs1
        if nrs1<num-1:        
            nrs1 +=1
        cnt_rs1 += 1    
    if rk==(max_rk-2):
        q_rs2[:,:,nrs2] = Q
        print 'Found rk=S-2, num = ', nrs2
        if nrs2<num-1:        
            nrs2 +=1
        cnt_rs2 += 1    
    if (cnt_rs>=num and cnt_rs1>=num and cnt_rs2>=num):
        not_full = False


prefix = "./data/Q/" + run + "_"

pickle.dump(q_rs, open(prefix+"q_rs.p", "wb"))
pickle.dump(q_rs1, open(prefix+"q_rs1.p", "wb"))
pickle.dump(q_rs2, open(prefix+"q_rs2.p", "wb"))

# -----------------------------------------------------------------------------#

num_species = 15
num_traits = 10
q_rs = np.zeros((num_species, num_traits, num))
q_rs1 = np.zeros((num_species, num_traits, num))
q_rs2 = np.zeros((num_species, num_traits, num))
run = 'S15_U10'

max_rk = np.min([num_species, num_traits])

not_full = True
nrs = 0
nrs1 = 0
nrs2 = 0
cnt_rs = 0
cnt_rs1 = 0
cnt_rs2 = 0
while not_full:
    Q, rk, s = generate_Q(num_species, num_traits)
    if rk==max_rk:
        q_rs[:,:,nrs] = Q
        print 'Found rk=S, num = ', nrs
        if nrs<num-1:
            nrs += 1
        cnt_rs += 1            
    if rk==(max_rk-1):
        q_rs1[:,:,nrs1] = Q
        print 'Found rk=S-1, num = ', nrs1
        if nrs1<num-1:        
            nrs1 +=1
        cnt_rs1 += 1    
    if rk==(max_rk-2):
        q_rs2[:,:,nrs2] = Q
        print 'Found rk=S-2, num = ', nrs2
        if nrs2<num-1:        
            nrs2 +=1
        cnt_rs2 += 1    
    if (cnt_rs>=num and cnt_rs1>=num and cnt_rs2>=num):
        not_full = False


prefix = "./data/Q/" + run + "_"

pickle.dump(q_rs, open(prefix+"q_rs.p", "wb"))
pickle.dump(q_rs1, open(prefix+"q_rs1.p", "wb"))
pickle.dump(q_rs2, open(prefix+"q_rs2.p", "wb"))
