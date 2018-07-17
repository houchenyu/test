# -*- coding:utf-8 -*-
###手写单隐层神经网络
import numpy as np

def sigmoid(X):
    return 1/(1+np.exp(-X))

def tanh(X):
    return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))


def set_layer(X, Y):
    n0 = X.shape[0]        ##输入层单元个数
    n1 = 4      ##隐藏层单元个数
    n2 = Y.shape[0]      ##输出层单元个数
    return (n0,n1,n2)

def initialization(n0, n1, n2):
    W1 = np.random.randn(n1, n0) * 0.01
    b1 = np.zeros((n1, 1))
    W2 = np.random.randn(n2, n1) * 0.01
    b2 = np.zeros((n2, 1))
    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }
    return  parameters

def forward_propagate(parameters, X):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1,X)+b1
    A1 = tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {
        'Z1':Z1,
        'A1':A1,
        'Z2':Z2,
        'A2':A2
    }
    return cache

def calculate_cost(A, Y):
    m = Y.shape[1]
    logloss = np.multiply(Y, np.log(A))+ np.multiply((1-Y), np.log(1-A))
    cost = -1/m * np.sum(logloss)
    return cost

def back_propagate(parameters, cache, X, Y, alpha):
    A1 = cache['A1']
    A2 = cache['A2']
    W1 = cache['W1']
    W2 = cache['W2']
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = 1.0/m* np.dot(dZ2, A1.T)
    db2 = 1.0/m*np.sum(dZ2, axis = 1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2)*(1- np.power(A1, 2))
    dW1 = 1.0/m* np.dot(dZ1, X.T)
    db1 = 1.0/m*np.sum(dZ1, axis = 1, keepdims=True)

    grads = {
        'dW2':dW2,
        'db2':db2,
        'dW1':dW1,
        'db1':db1
    }

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return grads,parameters

def train(X_train, Y_train, iter):
    n0, n1, n2 = set_layer(X_train, Y_train)
    parameters = initialization(n0, n1, n2)
    for i in range(iter):
        cache = forward_propagate(parameters, X_train)
        cost = calculate_cost(cache['A2'], Y_train)
        grads, parameters = back_propagate(parameters, cache, X_train, Y_train, 0.001)
        if i % 100 == 0:
            print('iter %d times, cost: %f' % (i, cost))

    return parameters

def test(parameters, X_test, Y_test):
    m = Y_test.shape[1]
    y_predict = np.zeros((1,m))
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']


    Z1 = np.dot(W1, X_test) + b1
    A1 = tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    for i in range(m):
        if A2[i] > 0.5:
            y_predict[i] = 1
        else:
            y_predict[i] = 0
    print("accuracy: {} %".format(100 - np.mean(np.abs(y_predict - Y_test)) * 100))
    return y_predict

if __name__ == '__main__':
    a = 1