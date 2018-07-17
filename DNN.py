# -*- utf-8 -*-
##手写 deep neural networks
import numpy as np

def sigmoid(X):
    return 1/(1+np.exp(-X))

def tanh(X):
    return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

def sigmoid_prime(X):
    return sigmoid(X)*(1-sigmoid(X))

def tanh_prime(X):
    return 1-np.power(tanh(X), 2)

def relu(X):
    return np.maximum(X, 0)

def relu_prime(X):
    s = X
    s[X<=0] = 0
    s[X>0] = 1
    return s

def initialize_parameters(layer):
    # Layer :[n0, n1, n2, ..., nL]
    np.random.seed(6)
    L = len(layer)
    parameters = {}
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.rand(layer[i], layer[i-1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layer[i], 1))
    return parameters

def calculate_cost(A, Y):
    m = Y.shape[1]
    logloss = np.multiply(Y, np.log(A)) + np.multiply(1-Y, np.log(1-A))
    cost = -1.0/m * np.sum(logloss)
    return cost

def train(X, Y, layer, iter, alpha ):
    m = Y.shape[1]
    L = len(layer)
    parameters = initialize_parameters(layer)
    fcache = {}   ## forward cache
    bcache = {}   ## backward cache
    costs = []
    fcache['A0'] = X
    for i in range(iter):
        # forward propagate
        for l in range(1, L):
            Zl = np.dot(parameters['W'+str(l)], fcache['A' + str(l)]) + parameters['b'+str(l)]
            Al = relu(Zl) if l != (L-1) else sigmoid(Zl)
            fcache['A'+str(l)] = Al
            fcache['Z'+str(l)] = Zl

        A = fcache['A'+str(L-1)]
        cost = calculate_cost(A, Y)

        # backward propagate
        L = L - 1
        bcache['dz' + str(L)] = A - Y
        bcache['dw' + str(L)] = 1.0/m*np.dot(bcache['dz' + str(L)], fcache['A' + str(L-1)].T)
        bcache['db' + str(L)] = 1.0/m*np.sum(bcache['dz' + str(L)], axis=1, keepdims=True)
        for l in reversed(range(1, L)):
            bcache['dz' + str(l)] = fcache['W' + str(l+1)].T, bcache['dz' + str(l+1)] * relu_prime(fcache['Z' + str(l)])
            bcache['dw' + str(l)] = (1.0/m) * np.dot(bcache['dz' + str(l)], fcache['A' + str(l-1)].T)
            bcache['db' + str(l)] = (1.0/m) * np.sum(bcache['dz' + str(l)], axis=1, keepdims=True)

        # update parameters
        for l in range(1, L+1):
            parameters['W' + str(l)] = parameters['W' + str(l)] - alpha * bcache['dw' + str(l)]
            parameters['b' + str(l)] = parameters['b' + str(l)] - alpha * bcache['db' + str(l)]

        if i % 100 == 0:
            print('cost after %d times , iteration is : %f' % (i, cost))
            costs = costs.append(cost)