import numpy as np
import pandas as pd


def sigmoid(z):
    '''
    Function: sigmoid
    Summary: Activation Function used for Logistic Regression and Neural Nets
    Attributes: 
        @param (z): scalar or numpy array of any size.
    Returns: outut of 1/(1+np.exp(-z))
    '''
    return 1 / (1 + np.exp(-z))


def logRegCostFxn(z, y):
    '''
    Function: logRegCostFxn
    Summary: InsertHere
    Attributes: 
        @param (z): output of final sigmoid activation fxn/neuron (y_pred)
        @param (y): actual output of sample 
    Returns: Cost function of logistic regression 
    '''
    return -(y * np.log10(z) + (1 - y) * np.log10(1 - z))



class logReg:
    '''
    Logistic regression
    '''

    def __init__(self):
        # J= Total cost of the costFXN
        self.J = 0
        # bias
        self.b = 0
        # weight
        self.w = None

    def fit(self, X, y, epoch=1000, learn_rate=0.01):

        # transpose the data so that (num_feat,num_training_samples)
        X = X.T
        self. w = np.zeros((X.shape[0], 1))  # model weight initialization

        for _ in range(epoch):
            m=X.shape[1]

            # output of the activation unit FORWARD PROPAGATION
            z = sigmoid(np.dot(self.w.T, X) + self.b)

            # compute the cost function
            self.J = 1 / m * np.sum(logRegCostFxn(z, y.T))

            # compute the derivatives for updating the weights and bias unit
            # BACK PROPAGATION
            dz = z - y.T
            dw = 1 / m * np.dot(X, dz.T)
            db = 1 / m * np.sum(dz)

            # updating the weights and bias unit based on the derivative of the
            # cost fxn
            self.b -= learn_rate * db
            self.w -= learn_rate * dw

        print('training has ended weights and bias are now fit for use in prediction')

    def test(self, X, y):
        
        X = X.T
        m=X.shape[1]

        z = sigmoid(np.dot(self.w.T, X) + self.b)
        
        y_prediction=[int(z[0,i] >= 0.5) for i in range(m)]

        return y_prediction



np.random.seed(100)
x = np.random.randn(50, 64, 64, 3)  # create a random dataset
y = np.random.rand(50, 1)  # output lab


# we are flipping to represent samples in the columns
x_flat = x.reshape(x.shape[0], -1)
# divide by the maximum value of a pixel channel
x_flat /= 255

jh=logReg()
jh.fit(x_flat,y)
print(jh.test(x_flat,y))



