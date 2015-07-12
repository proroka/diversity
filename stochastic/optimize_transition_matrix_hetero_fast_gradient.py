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
# helper functions to verify the gradient.

# Little helper to make nice looking terminal prints.
def hilite(string, success, failure, bold):
    attr = []
    if success:
        attr.append('32')  # green
    if not success and failure:
        attr.append('31')  # red
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def gradcheck_naive(f, x):
    fx, grad = f(x)  # Evaluate function value at original point
    assert x.shape == grad.shape, 'Variable and gradient must have the same shape'
    h = 1e-4
    passed = True
    numerical_grad = np.empty_like(x)

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        x[ix] -= h
        y1 = f(x)[0]
        x[ix] += 2 * h
        y2 = f(x)[0]
        numgrad = (y2 - y1) / (2 * h)
        x[ix] -= h
        numerical_grad[ix] = numgrad
        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5 and passed:
            # Only print the first error.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.ComplexWarning)  # Ignore complex warning.
                print hilite("Gradient check failed.", False, True, True)
                print "First gradient error found at index %s" % str(ix)
                print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            passed = False
        it.iternext()  # Step to next dimension
    if passed:
        print hilite("Gradient check passed!", True, False, True)
    return numerical_grad


# -----------------------------------------------------------------------------#
# auxiliary functions


# to give feedback during the optimization.
current_iteration = 0
def Print(x, f, accepted):
    global current_iteration
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

# Different optimization modes.
ABSOLUTE_AT_LEAST = 0
ABSOLUTE_EXACT = 1
QUADRATIC_AT_LEAST = 2
QUADRATIC_EXACT = 3

# In:
# desired_state: matrix with node-trait distribution
# elements: vector with transition matrix values for all K_i, speices i
# transform: species-trait matrix
# initial_state: node-species matrix
#
# If max_time is None, the optimization assumes that max_time is the last element
# in elements and the function will compute the gradient w.r.t. to max_time too.
# If optimize_t is larger than 0, the cost also minimizes (optimize_t * t^2).
# If force_steady_state > 0, the optimization will check the final state at max_time and max_time + force_steady_state.
def Cost_Fast(elements, desired_state, adjacency_matrix, max_time, initial_state, transform,
              cost_mode=QUADRATIC_EXACT, optimize_t=0.0, margin=0.0,
              force_steady_state=0.0):

    nstates = adjacency_matrix.shape[0]
    num_species = initial_state.shape[1]
    num_traits = transform.shape[1]
    # Renaming.
    Adj = adjacency_matrix.astype(float).reshape((nstates, nstates))
    if max_time is None:
        t = elements[-1]
        num_elements_i = (np.size(elements) - 1) / num_species
        grad_all = np.zeros(np.size(elements))
        elements = elements[:-1]
    else:
        optimize_t = 0.0  # Overwrite optimize_t if we are not optimizing t.
        t = max_time
        num_elements_i = np.size(elements) / num_species
        grad_all = np.zeros(np.size(elements))

    # Value (evaluate cost)
    # evaluate the value of the cost
    ExpAx0 = FinalState(elements, adjacency_matrix, t, initial_state, transform) - desired_state # difference
    if cost_mode == ABSOLUTE_AT_LEAST:
        ExpAx0 = -ExpAx0 + margin
        value = np.sum(np.maximum(ExpAx0, 0))
        ExpAx0 = -(ExpAx0 > 0).astype(float)  # Keep only 1s for when it's larger than margin.
    elif cost_mode == ABSOLUTE_EXACT:
        AbsExpAx0 = np.abs(ExpAx0)
        ZeroIndex = AbsExpAx0 < 1e-10
        value = np.sum(np.abs(ExpAx0))  # cost is sum of difference matrix absolutes.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # We don't care about 0/0.
            ExpAx0 = ExpAx0 / AbsExpAx0 # Keep only 1s for when it's larger than 0 and -1s for when it's lower.
        ExpAx0[ZeroIndex] = 0  # Make sure we set 0/0 to 0.
    elif cost_mode == QUADRATIC_AT_LEAST:
        ExpAx0 = -ExpAx0 + margin
        value = np.sum(np.square(np.maximum(ExpAx0, 0)))
        SmallerIndex = ExpAx0 < 0
        ExpAx0 *= -2.0
        ExpAx0[SmallerIndex] = 0  # Don't propagate gradient on smaller values.
    elif cost_mode == QUADRATIC_EXACT:
        value = np.sum(np.square(ExpAx0))  # cost is sum of difference matrix squared
        ExpAx0 *= 2.0
    value += optimize_t * (t ** 2) # add cost of t if we are optimizing for it

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
        x0m = np.dot(x0, m)
        top_grad = np.dot(ExpAx0, x0m.T)  # top_grad is the gradient w.r.t. ExpA of the inside of the parenthesis
        gradA = U.dot(V.T.dot(top_grad).dot(U) * X).dot(V.T)
        gradA2 = gradA * t
        grad = gradA2 - np.diag(gradA2)
        grad = grad.flatten()[Adj.flatten().astype(bool)]  # Reshape.

        grad_all[s*num_elements_i:(s+1)*num_elements_i] = np.array(np.real(grad))
        if max_time is None:
            grad_all[-1] += np.real(np.sum(A * gradA))

    if max_time is None:
        grad_all[-1] += 2.0 * t * optimize_t

    # Forcing the steady state (repeat with t = t + force_steady_state)
    # Copy-paste of the previous code.
    if force_steady_state and optimize_t and (max_time is None):
        t = t + force_steady_state
        # Value (evaluate cost)
        # evaluate the value of the cost
        ExpAx0 = FinalState(elements, adjacency_matrix, t, initial_state, transform) - desired_state # difference
        if cost_mode == ABSOLUTE_AT_LEAST:
            ExpAx0 = -ExpAx0 + margin
            value += np.sum(np.maximum(ExpAx0, 0))
            ExpAx0 = -(ExpAx0 > 0).astype(float)  # Keep only 1s for when it's larger than margin.
        elif cost_mode == ABSOLUTE_EXACT:
            AbsExpAx0 = np.abs(ExpAx0)
            ZeroIndex = AbsExpAx0 < 1e-10
            value += np.sum(np.abs(ExpAx0))  # cost is sum of difference matrix absolutes.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # We don't care about 0/0.
                ExpAx0 = ExpAx0 / AbsExpAx0 # Keep only 1s for when it's larger than 0 and -1s for when it's lower.
            ExpAx0[ZeroIndex] = 0  # Make sure we set 0/0 to 0.
        elif cost_mode == QUADRATIC_AT_LEAST:
            ExpAx0 = -ExpAx0 + margin
            value += np.sum(np.square(np.maximum(ExpAx0, 0)))
            SmallerIndex = ExpAx0 < 0
            ExpAx0 *= -2.0
            ExpAx0[SmallerIndex] = 0  # Don't propagate gradient on smaller values.
        elif cost_mode == QUADRATIC_EXACT:
            value += np.sum(np.square(ExpAx0))  # cost is sum of difference matrix squared
            ExpAx0 *= 2.0

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
            x0m = np.dot(x0, m)
            top_grad = np.dot(ExpAx0, x0m.T)  # top_grad is the gradient w.r.t. ExpA of the inside of the parenthesis
            gradA = U.dot(V.T.dot(top_grad).dot(U) * X).dot(V.T)
            gradA2 = gradA * t
            grad = gradA2 - np.diag(gradA2)
            grad = grad.flatten()[Adj.flatten().astype(bool)]  # Reshape.

            grad_all[s*num_elements_i:(s+1)*num_elements_i] += np.array(np.real(grad))
            grad_all[-1] += np.real(np.sum(A * gradA))


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
def ElementsBounds(nelements, max_rate, max_time, f_new, x_new, f_old, x_old):
    if max_time is None:
        elements_min = np.ones(nelements) * 0.  # min_rate.
        elements_max = np.ones(nelements) * max_rate
    else:
        elements_min = np.ones(nelements + 1) * 0.  # min_rate.
        elements_max = np.ones(nelements + 1) * max_rate
        elements_max[-1] = max_time
    return (bool(np.all(x_new <= elements_max)) and
            bool(np.all(x_new >= elements_min)))



# -----------------------------------------------------------------------------#
# optimization: basin hopping

def Optimize_Hetero_Fast(adjacency_matrix, initial_state, desired_steadystate, transform, max_time, max_rate, verbose):
    global current_iteration
    current_iteration = 0

    verify_gradient = False
    cost_mode = QUADRATIC_EXACT
    optimizing_t = False
    force_steady_state = 0.0 # 10.0  # Only used when optimizing_t is True. Set to zero if you don't care about steady-state.
    alpha = 0.1  # + alpha * t^2.
    mu = 0.0  # margin

    # initial random elements (only where the adjacency matrix has a 1).
    num_species = initial_state.shape[1]
    num_nonzero_elements = np.sum(adjacency_matrix)
    # assume that all species have same adjacency matrix
    init_elements = np.random.rand(num_nonzero_elements*num_species) * max_rate
    bounds = [(0., max_rate)] * num_nonzero_elements*num_species

    # Check gradient if requested.
    if verify_gradient:
        init_elements_with_time = np.concatenate([init_elements, np.array([max_time])], axis=0)
        # Randomly modify the init_elements and verify gradient 10 times.
        for i in range(10):
            new_init_elements = np.random.rand(*init_elements.shape)
            new_init_elements_with_time = np.random.rand(*init_elements_with_time.shape)
            margin = np.random.rand(1)
            optimize_t = np.random.rand(1)
            for mode in [ABSOLUTE_AT_LEAST, ABSOLUTE_EXACT, QUADRATIC_AT_LEAST, QUADRATIC_EXACT]:
                # Without optimizing t.
                gradcheck_naive(lambda x: Cost_Fast(x, desired_steadystate, adjacency_matrix, max_time, initial_state, transform,
                                                    cost_mode=mode, margin=margin, force_steady_state=force_steady_state), new_init_elements)
                # Optimizing t.
                gradcheck_naive(lambda x: Cost_Fast(x, desired_steadystate, adjacency_matrix, None, initial_state, transform,
                                                    cost_mode=mode, margin=margin, optimize_t=optimize_t, force_steady_state=force_steady_state), new_init_elements_with_time)

    # Prepare cost function.
    if optimizing_t:
        init_elements = np.concatenate([init_elements, np.array([max_time])], axis=0)
        bounds.append((0., max_time * 2))  # Allow optimized time be up to twice as large as max_time.
        CostFunction = lambda x: Cost_Fast(x, desired_steadystate, adjacency_matrix, None, initial_state, transform, cost_mode=cost_mode, optimize_t=alpha, margin=mu, force_steady_state=force_steady_state)
        BoundFunction = lambda f_new, x_new, f_old, x_old: ElementsBounds(num_nonzero_elements*num_species, max_rate, 2.0 * max_time, f_new, x_new, f_old, x_old)
    else:
        CostFunction = lambda x: Cost_Fast(x, desired_steadystate, adjacency_matrix, max_time, initial_state, transform, cost_mode=cost_mode, margin=mu, force_steady_state=force_steady_state)
        BoundFunction = lambda f_new, x_new, f_old, x_old: ElementsBounds(num_nonzero_elements*num_species, max_rate, None, f_new, x_new, f_old, x_old)

    # basinhopping function
    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'jac': True, 'options': {'disp': False}}
    ret = scipy.optimize.basinhopping(CostFunction,
                                      init_elements,
                                      minimizer_kwargs=minimizer_kwargs,
                                      niter=100, niter_success=3,
                                      accept_test=BoundFunction,
                                      callback=None if not verbose else Print)

    # Remove the optimized t.
    if optimizing_t:
        optimal_t = ret.x[-1]
        ret.x = ret.x[:-1]
        ret.fun -= alpha * (optimal_t ** 2)
        if force_steady_state:
            ret.fun /= 2.0
    else:
        optimal_t = max_time

    if verbose:
        print '\n\nFinal matrix:\n', MatrixReshape(ret.x, adjacency_matrix, num_species)
        print '\nSum columns:', np.sum(MatrixReshape(ret.x, adjacency_matrix, num_species), axis=0)
        print '\nReaches:\n', FinalState(ret.x, adjacency_matrix, optimal_t, initial_state, transform)
        print '\nError (at time %.2f):\n' % optimal_t, FinalState(ret.x, adjacency_matrix, optimal_t, initial_state, transform) - desired_steadystate
        if optimizing_t:
            print '\nOptimal time:', optimal_t
            print '\nError (at max time %.2f):\n' % max_time, FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform) - desired_steadystate


    # print this always
    print '\nFinal cost (ignoring cost on time):\n', ret.fun

    # return transition matrices (3D matrix for all species)
    return MatrixReshape(ret.x, adjacency_matrix, num_species)

      ####







