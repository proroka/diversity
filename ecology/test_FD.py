# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 15:30:44 2015

@author: amandaprorok
"""

import numpy as NP
import pylab as PL
import matplotlib.pyplot as PP
import matplotlib.image as IM
import scipy.cluster.hierarchy as CH
# my modules
import funcdef_diversity as FD

# species and traits
num_species = 10
num_traits = 2
p_species = NP.random.rand(num_species,1) 
p_species = p_species / NP.sum(p_species) # normalize abundance
traits = NP.zeros((num_traits,num_species))
# initialize trait values in interval [0,1]
for i in range(num_traits):
    traits[i,:] = NP.random.rand(1,num_species)


# plot species with pint size proportional to abundance
PP.figure()
PL.xlabel('$t_1$')
PL.ylabel('$t_2$')
colors = NP.random.rand(num_species)
f = 12. / NP.max(p_species)
area = NP.pi * (p_species*f)**2 # 0 to 12 point radiuses
PP.scatter(traits[0,:],traits[1,:],s=area,c=colors,alpha=0.6)
PP.xlim(0,1)
PP.ylim(0,1)
PP.axis('equal')
PL.show()

# sum of dendrogram branches
dist_species_v = FD.distance_v(traits)
PP.figure()
Z = CH.linkage(dist_species_v,method='single',metric='euclidean')
dend = CH.dendrogram(Z)
b = FD.branch_lengths(Z)
h = FD.branch_presence(Z)
div_fd = FD.fd(h,b)

# calculate rao's Q
dist_species_m = FD.distance_m(traits)
div_q = FD.rao(dist_species_m,p_species)

# calculate FAD
div_fad = FD.fad(dist_species_m)

# print diversity values
print "Functional Diversity: ", div_fd
print "Rao's Q: ", div_q
print "Functional Attribute Diversity: ", div_fad






