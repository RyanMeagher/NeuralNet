import numpy as np
import pandas as pd
import utilityFxns


def computeCost(y_pred, y):
    m = y.shape[1]

    J = (-1 / m) * np.sum(np.multiply(y, np.log(y_pred)) +
                          np.multiply((1 - y), np.log(1 - y_pred)))

    return J


class NNet:

    def __init__(self):

        self.layers = []
        self.biases = []
        self.activationFxns = []
        self.num_classes = 0
        self.shape = {}
        self.count = 0

    def firstLayer(self, X, num_neurons=10, activationFxn=np.tanh):

        n_x = X.shape[0]

        b = np.zeros((num_neurons, 1))
        w = np.random.randn(num_neurons, n_x) * 0.01

        self.biases.append(b)
        self.layers.append(w)
        self.activationFxns.append(activationFxn)
        self.shape['input layer'] = n_x
        self.count += 1
        self.shape['hiddenLayer' + str(self.count)] = num_neurons

        return

    def hiddenLayer(self, num_neurons=10, activationFxn=np.tanh):
        w = np.random.randn(num_neurons, self.layers[-1].shape[0]) * 0.01
        b = np.zeros((num_neurons, 1))

        self.layers.append(w)
        self.biases.append(b)
        self.activationFxns.append(activationFxn)
        self.count += 1
        self.shape['hiddenLayer' + str(self.count)] = num_neurons

        return

    def finalLayer(self, num_classes=2, activationFxn=utilityFxns.sigmoid):
        if num_classes == 2:

            w = np.random.randn(1, self.layers[-1].shape[0]) * 0.01
            b = np.zeros((1, 1))
        else:
            w = np.random.randn(num_classes, self.layers[-1].shape[0]) * 0.01
            b = np.zeros((num_classes, 1))

        self.layers.append(w)
        self.biases.append(b)
        self.activationFxns.append(activationFxn)
        self.count+=1
        self.num_classes = num_classes
        self.shape['FinalLayer'] = self.num_classes


        return

    def ForwardPropagate(self, X):
        cache=[]

        y_pred = X
        for i in range(len(self.layers)):
            y_pred = np.dot(self.layers[i], y_pred) + self.biases[i]

            y_pred = self.activationFxns[i](y_pred)
            cache.append(y_pred)


        return y_pred, cache

    def BackProp(self, cache, X, y_pred, y, learning_rate=0.01):
        dw=[]
        dz=[]
        db=[]
        m=cache[0].shape[1]

        for i in reversed(range(self.count)):
            if i==self.count-1:
                dz=y_pred - y
                dw=1 / m * np.dot(cache[i], dz.T)
                db=1 / m * np.sum(dz, axis=1, keepdims=True)
                self.layers[i] -= learning_rate*dw 
                self.biases[i] -= learning_rate*db 
            else:
                pass
                #dz=np.dot(self.layers[i+1].T,dz) * (1-np.power(self.layers[i],2))







        



np.random.seed(1)
X = np.random.randn(2, 400)
Y = np.random.randn(1, 400)


net = NNet()

net.firstLayer(X)
net.hiddenLayer(15)
net.finalLayer()

y_pred,cache=net.ForwardPropagate(X)

print(len(cache))
net.BackProp(cache, X, y_pred, Y)





