# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:35:05 2015
@author: amanda

"""

import numpy as np
import scipy
import scipy.linalg
import networkx as nx
import random
import warnings


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


# Gradient checker.
# Returns the true gradient.
def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """
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


# || Exp(A * t) * x0 - xd ||^2
# Returns the value and the gradient w.r.t. A
def func1(A, t, x0, xd):
    # Value.
    ExpA = scipy.linalg.expm(A * t)
    ExpAx0 = ExpA.dot(x0) - xd
    value = ExpAx0.T.dot(ExpAx0)

    # Gradient preparation.
    w, V = scipy.linalg.eig(A * t, right=True)
    U = scipy.linalg.inv(V).T
    exp_w = np.exp(w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # We don't care about 0/0 on the diagonal.
        X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
    np.fill_diagonal(X, exp_w)
    # Actual gradient.
    top_grad = 2 * ExpAx0.dot(x0.T)  # Gradients from || e^At * x - xd ||^2 w.r.t to e^At is 2 * (e^At * x - xd) * xT
    grad = U.dot(V.T.dot(top_grad).dot(U) * X).dot(V.T)
    grad = grad * t

    return [value, np.real(grad)]


# || Exp(A * t) * x0 - xd ||^2
# Returns the value and the gradient w.r.t. t
# Most of this code is the same as the function func1 above, the only difference is in the final gradient line.
def func2(A, t, x0, xd):
    # Value.
    ExpA = scipy.linalg.expm(A * t)
    ExpAx0 = ExpA.dot(x0) - xd
    value = ExpAx0.T.dot(ExpAx0)

    # Gradient preparation.
    w, V = scipy.linalg.eig(A * t, right=True)
    U = scipy.linalg.inv(V).T
    exp_w = np.exp(w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # We don't care about 0/0 on the diagonal.
        X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
    np.fill_diagonal(X, exp_w)
    # Actual gradient.
    top_grad = 2 * ExpAx0.dot(x0.T)  # Gradients from || e^At * x - xd ||^2 w.r.t to e^At = 2 * (e^At * x - xd) * xT
    grad = U.dot(V.T.dot(top_grad).dot(U) * X).dot(V.T)
    grad = np.matrix(np.sum(A * grad))  # Only different with above.

    return [value, np.real(grad)]


# || Exp(F(params) * t) * x0 - xd ||^2
# Returns the value and the gradient w.r.t. params
# Most of this code is the same as the function func1 above, the only difference is in the end.
def func3(params, t, x0, xd, Adj):
    # Create A from w.
    # We assume params are off-diagonal values corresponding to the adjacency matrix adj.
    A = np.zeros_like(Adj.flatten())
    A[Adj.flatten().astype(bool)] = params
    A = A.reshape(Adj.shape)
    np.fill_diagonal(A, -np.sum(A, axis=0))

    # Value.
    ExpA = scipy.linalg.expm(A * t)
    ExpAx0 = ExpA.dot(x0) - xd
    value = ExpAx0.T.dot(ExpAx0)

    # Gradient preparation.
    w, V = scipy.linalg.eig(A * t, right=True)
    U = scipy.linalg.inv(V).T
    exp_w = np.exp(w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # We don't care about 0/0 on the diagonal.
        X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
    np.fill_diagonal(X, exp_w)
    # Gradient w.r.t. A.
    top_grad = 2 * ExpAx0.dot(x0.T)  # Gradients from || e^At * x - xd ||^2 w.r.t to e^At = 2 * (e^At * x - xd) * xT
    gradA = U.dot(V.T.dot(top_grad).dot(U) * X).dot(V.T)
    gradA = gradA * t
    # Actual gradient.
    # Compute gradient w.r.t. params.
    # Note that using Numpy broadcasting, doing
    # [ 1 1 1 ]             [ 2 3 4 ]
    # [ 1 1 1 ] + [1 2 3] = [ 2 3 4 ], i.e., each column is incremented by the corresponding column of the vector.
    # [ 1 1 1 ]             [ 2 3 4 ]
    grad = gradA - np.diag(gradA)
    grad = grad.flatten()[Adj.flatten().astype(bool)]  # Reshape.

    return [value, np.real(grad)]


########################
#       Test code      #
########################

# Matrix size.
N = 4

# Random values for all variables.
# Gradient will be computed at these locations.
A = np.random.rand(N, N)
t = np.random.rand(1, 1)
x0 = np.random.rand(N, 1)
xd = np.random.rand(N, 1)

# Only used for func3.
# Create random adjacency matrix using networkx.
assert N >= 2, 'N must be larger or equal to 2'
if N > 2:
    G = nx.connected_watts_strogatz_graph(N, N - 1, 0.5)
    Adj = nx.to_numpy_matrix(G)
    Adj = np.squeeze(np.asarray(Adj))
else:
    # Fully connected.
    Adj = np.ones((N, N))
    np.fill_diagonal(Adj, 0)
params = np.random.rand(np.sum(Adj))  # Pick random transition rates.

print '\nValue t =\n', t, '\n'
print 'Vector x0 =\n', x0, '\n'
print 'Vector xd =\n', xd, '\n'
print 'Matrix A =\n', A, '\n'

# Gradient with respect to A.
print '\n-------------------------------------'
print hilite('Trying to compute the gradient of ||exp(A * t) * x0 - xd||^2 w.r.t A\n', False, False, True)
value, grad = func1(A, t, x0, xd)
print '||exp(A * t) * x0||^2 =', value[0][0], '\n'
print 'd{||exp(A * t) * x0 - xd||^2} / dA =\n', grad, '\n'
print 'Checking gradient:'
numgrad = gradcheck_naive(lambda x: func1(x, t, x0, xd), A)
print '\nNumerical gradient is\n', numgrad, '\n'

# Gradient with respect to t.
print '\n-------------------------------------'
print hilite('Trying to compute the gradient of ||exp(A * t) * x0 - xd||^2 w.r.t t\n', False, False, True)
value, grad = func2(A, t, x0, xd)
print '||exp(A * t) * x0||^2 =', value[0][0], '\n'
print 'd{||exp(A * t) * x0 - xd||^2} / dt =\n', grad, '\n'
print 'Checking gradient:'
numgrad = gradcheck_naive(lambda x: func2(A, x, x0, xd), t)
print '\nNumerical gradient is\n', numgrad, '\n'

# Gradient with respect to w.
print '\n-------------------------------------'
print '\nArray params =\n', params, '\n'
print 'Adjacency =\n', Adj, '\n'
print hilite('Trying to compute the gradient of ||exp(F(w) * t) * x0 - xd||^2 w.r.t w\n', False, False, True)
value, grad = func3(params, t, x0, xd, Adj)
print '||exp(F(w) * t) * x0||^2 =', value[0][0], '\n'
print 'd{||exp(F(w) * t) * x0 - xd||^2} / dw =\n', grad, '\n'
print 'Checking gradient:'
numgrad = gradcheck_naive(lambda x: func3(x, t, x0, xd, Adj), params)
print '\nNumerical gradient is\n', numgrad, '\n'
