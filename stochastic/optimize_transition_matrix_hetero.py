# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:39:47 2015
@author: amandaprorok

"""

import scipy
import scipy.optimize
import scipy.linalg
import numpy as np
import sys


# -----------------------------------------------------------------------------#
# auxiliary functions
    

# To give feedback during the optimization.
def Print(x, f, accepted, current_iteration):
    current_iteration += 1
    sys.stdout.write('\r%i) Cost: %.4f, Accepted %s' % (current_iteration, f, str(accepted)))
    sys.stdout.flush()

# Computes the state after max_time.
# elements is array: flattened k_ij matrix, flattened for all species
def FinalState(elements, adjacency_matrix, max_time, initial_state, transform):
    num_nodes = initial_state.shape[0]    
    num_species = initial_state.shape[1]   
    deploy_robots_final = np.zeros((num_nodes, num_species))    

    K_all = MatrixReshape(elements, adjacency_matrix, num_species)

    for s in range(num_species):
        K = K_all[:,:,s]
        deploy_robots_final[:,s] = np.dot(scipy.linalg.expm(K * max_time), initial_state[:,s])
    # make deployment matrix with nodes x species (columns are xi)
    deploy_traits_final = np.dot(deploy_robots_final, transform)
    return deploy_traits_final


# Defines the cost to minimize.
# elements is array of nonzero elements for each species
def Cost(elements, desired_state, adjacency_matrix, max_time, initial_state, transform):
    #return np.mean(np.square(desired_state - FinalState(elements, adjacency_matrix, max_time, initial_state, transform))
    diffsq = np.square(desired_state - FinalState(elements, adjacency_matrix, max_time, initial_state, transform))  
    return np.sum(np.sum(diffsq))


# Reshapes the array of nonzero elements of the matrix into a valid transition matrix.
def MatrixReshape(elements_array, adjacency_matrix, num_species):
    nstates = adjacency_matrix.shape[0]
    # Place elements where the adjacency matrix has value 1.
    adjacency_matrix = adjacency_matrix.astype(bool)
    a = adjacency_matrix.flatten()
    K_all = np.zeros((nstates, nstates, num_species))
    num_nodes = adjacency_matrix.shape[0]    
    num_elements = elements_array.shape[0] / num_species
    for s in range(num_species):
        matrix_elements = np.zeros(nstates ** 2)
        elements = elements_array[s*num_elements:(s+1)*num_elements] 
        matrix_elements[a] = elements
        K = np.array(matrix_elements).reshape(num_nodes, num_nodes)
        np.fill_diagonal(K, -np.sum(K, axis=0))
        K_all[:,:,s] = K        
    return K_all
    

def ElementsBounds(nelements, max_rate, f_new, x_new, f_old, x_old):
   elements_min = np.ones(nelements) * 0.  # min_rate.
   elements_max = np.ones(nelements) * max_rate
   return (bool(np.all(x_new <= elements_max)) and
           bool(np.all(x_new >= elements_min)))
     

              
# -----------------------------------------------------------------------------#
# optimization

# create random transition matrix
# transition_m = np.zeros((num_nodes,num_nodes,num_species))

# transition_m = Optimize_Hetero(adjacency_m, deploy_robots_init, deploy_traits_desired, species_traits, max_time, max_rate, verbose=True)

def Optimize_Hetero(adjacency_matrix, initial_state, desired_steadystate, transform, max_time, max_rate, verbose):
    current_iteration = 0
    
    
    # Initial random elements (only where the adjacency matrix has a 1).
    num_species = initial_state.shape[1]
    num_nonzero_elements = np.sum(adjacency_matrix)
    # assume that all species have same adjacency matrix
    init_elements = np.random.rand(num_nonzero_elements*num_species) * max_rate
    
    
    bounds = [(0., max_rate)] * num_nonzero_elements*num_species
    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds} # L-BFGS-B is the one which takes bounds
    
    # basinhop optimization
    ret = scipy.optimize.basinhopping(lambda x: Cost(x, desired_steadystate, adjacency_matrix, max_time, initial_state, transform), 
                                      init_elements, 
                                      minimizer_kwargs=minimizer_kwargs, 
                                      niter=10,
                                      accept_test=lambda f_new, x_new, f_old, x_old: ElementsBounds(num_nonzero_elements*num_species, max_rate, f_new, x_new, f_old, x_old),                                      
                                      callback=None if not verbose else lambda x, f, accept: Print(x, f, accept, current_iteration))
                                      
    
    if verbose:
        print '\n\nFinal matrix:\n', MatrixReshape(ret.x, adjacency_matrix, num_species)
        print '\nSum columns:', np.sum(MatrixReshape(ret.x, adjacency_matrix, num_species), axis=0)
        print '\nFinal cost:\n', ret.fun
        print '\nReaches:\n', FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform)
        print '\nError:\n', FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform) - desired_steadystate
        
    return MatrixReshape(ret.x, adjacency_matrix, num_species)
      
      