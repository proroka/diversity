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
import warnings

# -----------------------------------------------------------------------------#
# auxiliary functions


# to give feedback during the optimization.
def Print(x, f, accepted, current_iteration):
    current_iteration += 1
    sys.stdout.write('\r%i) Cost: %.4f, Accepted %s' % (current_iteration, f, str(accepted)))
    sys.stdout.flush()

# computes the state after max_time.
# elements is array: flattened k_ij matrix, flattened for all species
def FinalState(elements, adjacency_matrix, max_time, initial_state, transform):
    num_nodes = initial_state.shape[0]
    num_species = initial_state.shape[1]
    deploy_robots_final = np.zeros((num_nodes, num_species))

    K_all = MatrixReshape(elements, adjacency_matrix, num_species)
    # for each species, get transition matrix and calculate final state
    for s in range(num_species):
        K = K_all[:,:,s]
        deploy_robots_final[:,s] = np.dot(scipy.linalg.expm(K * max_time), initial_state[:,s])
    # transform species final state to traits final state
    deploy_traits_final = np.dot(deploy_robots_final, transform)
    return deploy_traits_final

# -----------------------------------------------------------------------------#

# defines the cost to minimize.
# parameter elements is array - concatenation of nonzero elements for each species
def Cost_OLD(elements, desired_state, adjacency_matrix, max_time, initial_state, transform):
    #return np.mean(np.square(desired_state - FinalState(elements, adjacency_matrix, max_time, initial_state, transform))
    diffsq = np.square(desired_state - FinalState(elements, adjacency_matrix, max_time, initial_state, transform))
    return np.sum(np.sum(diffsq))

# -----------------------------------------------------------------------------#

# In:
# desired_state: matrix with node-trait distribution
# elements: vector with transition matrix values for all K_i, speices i
# transform: species-trait matrix
# initial_state: node-species matrix
def Cost_Fast(elements, desired_state, adjacency_matrix, max_time, initial_state, transform):

    nstates = adjacency_matrix.shape[0]
    num_species = initial_state.shape[1]
    num_traits = transform.shape[1]
    # Renaming.
    Adj = adjacency_matrix.astype(float).reshape((nstates, nstates))
    t = max_time
    num_elements_i = np.size(elements) / num_species
    grad_all = np.zeros(np.size(elements))

    # Value (evaluate cost)
    # evaluate the value of the cost
    ExpAx0 = FinalState(elements, adjacency_matrix, max_time, initial_state, transform) - desired_state # difference
    value = np.sum(np.square(ExpAx0)) # cost is sum of difference matrix squared

    # Calculate gradient w.r.t. transition matrix for each species
    for s in range(num_species):
        x0 = initial_state[:,s].reshape((nstates,1))
        m = transform[s,:].reshape((1,num_traits))
        el = elements[s*num_elements_i:(s+1)*num_elements_i]

        # Create A from elements.
        # We assume w are off-diagonal values corresponding to the adjacency matrix adj.
        A = np.zeros_like(Adj.flatten())
        A[Adj.flatten().astype(bool)] = el
        A = A.reshape(Adj.shape)
        np.fill_diagonal(A, -np.sum(A, axis=0))

        # Gradient preparation.
        w, V = scipy.linalg.eig(A * t, right=True)
        U = scipy.linalg.inv(V).T
        exp_w = np.exp(w)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", RuntimeWarning)  # We don't care about 0/0 on the diagonal.
          X = np.subtract.outer(exp_w, exp_w) / (np.subtract.outer(w, w) + 1e-10)
        np.fill_diagonal(X, exp_w)
        # Gradient w.r.t. A.
        #top_grad = 2 * ExpAx0.dot(x0.T)  # Gradients from || e^At * x - xd ||^2 w.r.t to e^At = 2 * (e^At * x - xd) * xT
        x0m = np.dot(x0, m)
        top_grad = 2 * np.dot(ExpAx0, x0m.T)
        gradA = U.dot(V.T.dot(top_grad).dot(U) * X).dot(V.T)
        gradA = gradA * t
        grad = gradA - np.diag(gradA)
        grad = grad.flatten()[Adj.flatten().astype(bool)]  # Reshape.

        grad_all[s*num_elements_i:(s+1)*num_elements_i] = np.array(np.real(grad))

    return [value, grad_all]



# -----------------------------------------------------------------------------#

# reshapes the array of nonzero elements of the matrix into a valid transition matrix.
# returns 3D matrix, 3rd dim indexes the species
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

# defines bounds on transition matrix values
def ElementsBounds(nelements, max_rate, f_new, x_new, f_old, x_old):
   elements_min = np.ones(nelements) * 0.  # min_rate.
   elements_max = np.ones(nelements) * max_rate
   return (bool(np.all(x_new <= elements_max)) and
           bool(np.all(x_new >= elements_min)))



# -----------------------------------------------------------------------------#
# optimization: basin hopping

def Optimize_Hetero_Fast(adjacency_matrix, initial_state, desired_steadystate, transform, max_time, max_rate, verbose):
    current_iteration = 0


    # initial random elements (only where the adjacency matrix has a 1).
    num_species = initial_state.shape[1]
    num_nonzero_elements = np.sum(adjacency_matrix)
    # assume that all species have same adjacency matrix
    init_elements = np.random.rand(num_nonzero_elements*num_species) * max_rate

    bounds = [(0., max_rate)] * num_nonzero_elements*num_species
    # minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds} # L-BFGS-B is the one which takes bounds
    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'jac': True, 'options': {'disp': False}}


    # basinhopping function
    ret = scipy.optimize.basinhopping(lambda x: Cost_Fast(x, desired_steadystate, adjacency_matrix, max_time, initial_state, transform),
                                      init_elements,
                                      minimizer_kwargs=minimizer_kwargs,
                                      niter=100, niter_success=3,
                                      accept_test=lambda f_new, x_new, f_old, x_old: ElementsBounds(num_nonzero_elements*num_species, max_rate, f_new, x_new, f_old, x_old),
                                      callback=None if not verbose else lambda x, f, accept: Print(x, f, accept, current_iteration))


    if verbose:
        print '\n\nFinal matrix:\n', MatrixReshape(ret.x, adjacency_matrix, num_species)
        print '\nSum columns:', np.sum(MatrixReshape(ret.x, adjacency_matrix, num_species), axis=0)
        print '\nFinal cost:\n', ret.fun
        print '\nReaches:\n', FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform)
        print '\nError:\n', FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform) - desired_steadystate

    # return transition matrices (3D matrix for all species)
    return MatrixReshape(ret.x, adjacency_matrix, num_species)

      ####











