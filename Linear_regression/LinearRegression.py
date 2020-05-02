# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import random
import numpy as np

class linearRegression():
    #y = k*x + b
    def __init__(self, x, y, lr=0.001):
        self.x = x
        self.y = y
        self.lr = lr
        self.k = np.random.normal()
        self.b = np.random.normal()

    def _calc_gradient(self):
        d_k = np.mean((self.x * self.k + self.b - self.y) * self.x)
        d_b = np.mean(self.x * self.k + self.b - self.y)
        return d_k, d_b

    def predict(self, x):
        y_predict = x * self.k + self.b
        return y_predict

    def loss(self):
        y_predict = self.predict(self.x)
        error = np.mean((self.y - y_predict)**2)
        return error

    def step(self):
        d_k, d_b = self._calc_gradient()
        self.k = self.k - self.lr * d_k
        self.b = self.b - self.lr * d_b
        return self.loss()

if  __name__ == "__main__":
    #Init data
    data_size = 50
    X_set = np.random.uniform(0, 10, data_size)
    Y_set = X_set * (-10) + 10 + np.random.normal(loc = -5.0, scale = 5.0, size = data_size)

    #plot data
    fig = plt.figure()
    resultPlot = fig.add_subplot(121)
    plt.title("Result")
    plt.xlabel("X")
    plt.ylabel("Y")
    resultPlot.scatter(X_set,Y_set)
    iterationsPlot = fig.add_subplot(122)
    plt.title("Iteriation")
    plt.xlabel("num of Iteriation")
    plt.ylabel("Loss")

    #linear Regression iteration
    MaxIter = 100
    MinLoss = 1
    LR = linearRegression(X_set, Y_set)
    iter = 0
    while (iter < MaxIter):
        loss = LR.step()
        if loss < MinLoss:
            break
        iterationsPlot.scatter(iter,loss,s= 10)
        iter = iter + 1
        plt.pause(0.01)

    #show result
    x = np.arange(min(X_set), max(X_set))
    y = LR.predict(x)
    resultPlot.plot(x,y,c = "red")
    resultPlot.text(5, -20, "y = %.2fx + %.2f" % (LR.k, LR.b), c = "r")
    plt.ioff() #turn off plot interaction
    plt.show()