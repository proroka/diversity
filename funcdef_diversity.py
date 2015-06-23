# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 15:30:44 2015

@author: amandaprorok
"""
import numpy as NP
import pylab as PL
import matplotlib.pyplot as PP
import matplotlib.image as IM
import numpy.linalg as LA


# ----------------------------------------------------------------#
# calculate distance metrics

# calculates a distance vector, Euclidean metric
# input: traits(rows)-species(columns) matrix
def distance_v(S):
    ns = NP.size(S,1)
    ind = 0;
    #D = NP.zeros((ns,ns))
    D = NP.zeros(ns*(ns-1)/2)
    for i in range(ns):
        j0 = i+1
        for j in range(j0,ns):
            ds = S[:,i]-S[:,j]
            D[ind] = LA.norm(ds);
            ind = ind+1;
    return D

# calculates a distance matrix, Euclidean metric
# input: traits(rows)-species(columns) matrix
def distance_m(S):
    ns = NP.size(S,1)
    D = NP.zeros((ns,ns))
    for i in range(ns):
        for j in range(ns):
            ds = S[:,i]-S[:,j]
            D[i,j] = LA.norm(ds);
    return D
    
# ----------------------------------------------------------------#
# Rao's quadratic entropy, Rao 1982
    
# input: distance matrix and abundance vector
def rao(D,P):
    q = 0
    for i in range(len(D)):
        for j in range(len(D)):
            q=q+D[i,j]*P[i]*P[j]
    return q
    

# ----------------------------------------------------------------#
# functional attribute diverisity, Walker 1999 

# input: distance matrix
def fad(D):
    q = 0
    for i in range(len(D)):
        for j in range(len(D)):
            q=q+D[i,j]
    return q


# ----------------------------------------------------------------#
# functional diversity, Petchey 2002

# input: linkage matrix: x-1 x 4
def branch_lengths(Z):
    x = NP.size(Z,0) + 1 # number of species
    B = NP.zeros(((x-1)*2,1)) # branch lengths

    # get the branch length vector
    n = NP.size(Z,0) # x-1
    ind = 0
    for i in range(n):
        # check if leaf node
        for j in range(2):
            if Z[i,j]<n:
                B[ind] = Z[i,2]
            else:
                t = Z[i,j]-(n+1)
                B[ind] = Z[i,2]-Z[t,2]
            ind += 1         
    return B

# input: linkage matrix: x-1 x 4
def branch_presence(Z):
    x = NP.size(Z,0) + 1 # number of species
    H = NP.zeros((x,(x-1)*2)) # branch presence matrix

    # create branch presence/absence matrix
    for ni in range(x): # for all leaf nodes
        np = ni # parent node    
        while(np<(x-1)*2): # root not reached
            for i in range(x-1): # rows of Z
                for j in range(2):
                    if Z[i,j]==np:
                        ind = i*2 + j # branch index
                        H[ni,ind] = 1
                        np = i+x # parent node                 
    return H
  
# calculate FD (Petchey 2002)                  
# input: H, branch presence matrix; B, branch length matrix
def fd(H,B):                  
    i = NP.sum(H,axis=0)
    FD = i.dot(B)
    return FD   
    

"""
doesnt work...
def check_branch(Z,H,i):
    # check if leaf node
    for j in range(2):
        print "i,j ", i,j
        nv = Z[i,j]
        ind = i*2 + j
        if nv<n: # is leaf node
            H[nv,ind] = 1
        else:
            t = Z[i,j]-(n+1)
            check_branch(Z,H,t)
    return       
            
check_branch(Z,H,n-1)            
"""   
    
