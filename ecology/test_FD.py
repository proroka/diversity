# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 15:30:44 2015

@author: amandaprorok
"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as pp
import matplotlib.image as im
import scipy.cluster.hierarchy as ch
# my modules
import funcdef_diversity as fd

# species and traits
num_species = 10
num_traits = 2
p_species = np.random.rand(num_species,1) 
p_species = p_species / np.sum(p_species) # normalize abundance
traits = np.zeros((num_traits,num_species))
# initialize trait values in interval [0,1]
for i in range(num_traits):
    traits[i,:] = np.random.rand(1,num_species)


# plot species with pint size proportional to abundance
pp.figure()
pl.xlabel('$t_1$')
pl.ylabel('$t_2$')
colors = np.random.rand(num_species)
f = 12. / np.max(p_species)
area = np.pi * (p_species*f)**2 # 0 to 12 point radiuses
pp.scatter(traits[0,:],traits[1,:],s=area,c=colors,alpha=0.6)
pp.xlim(0,1)
pp.ylim(0,1)
pp.axis('equal')
pl.show()

# sum of dendrogram branches
dist_species_v = fd.distance_v(traits)
pp.figure()
Z = ch.linkage(dist_species_v,method='single',metric='euclidean')
dend = ch.dendrogram(Z)
b = fd.branch_lengths(Z)
h = fd.branch_presence(Z)
div_fd = fd.fd(h,b)

# calculate rao's Q
dist_species_m = fd.distance_m(traits)
div_q = fd.rao(dist_species_m,p_species)

# calculate FAD
div_fad = fd.fad(dist_species_m)

# print diversity values
print "Functional Diversity: ", div_fd
print "Rao's Q: ", div_q
print "Functional Attribute Diversity: ", div_fad






