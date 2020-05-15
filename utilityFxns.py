import numpy as np


def sigmoid(z):
    '''
    Function: sigmoid
    Summary: Activation Function used for Logistic Regression and Neural Nets
    Attributes:
        @param (z): scalar or numpy array of any size.
    Returns: outut of 1/(1+np.exp(-z))
    '''
    return 1 / (1 + np.exp(-z))


def Relu(z):
    return np.max(0, z)


def leakyRelu(z, minimization_param=0.01):
    return np.max(minimization_param * z, z)


