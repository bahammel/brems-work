import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import torch


Z = 1.
# ne = 1.
# ni = ne


def brems(ne, kTe, Z, x):
    kTe *= 1.e3
    y = 1.e40 * 1.e-5 * 5.34e-39 * Z**2. * (ne)**2.* (1.6e-12 * kTe)**-0.5 * np.exp(-x/kTe)    
    return y


def get_data_3():
    kTe = np.linspace(1, 6, 2)
    ne = np.linspace(1, 3, 2)
    x = np.linspace(1000, 5000, 81)

    X = []
    Y = []

    for t in kTe:
        for k in ne:
            for i in x:
                Y.append([t,k])
                X.append([i, brems(k, t, Z, i)])

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y)
    return map(np.asarray, [xtrain, xtest, ytrain, ytest])


def get_data_2():
    kTe = np.linspace(1, 6, 2)
    x = np.linspace(1000, 5000, 51)

    X = []
    Y = []

    for t in kTe:
        for i in x:
            Y.append([t])
            X.append([i, brems(ne, t, Z, i)])

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

    return map(np.asarray, [xtrain, xtest, ytrain, ytest])

def get_data():
    kTe = np.linspace(1, 5, 5)
    x = np.linspace(1000, 5100, 1000)

    X = []
    Y = []

    for t in kTe:
        for i in x:
            X.append([i, t])
            Y.append([brems(ne, t, Z, i)])

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

    return map(np.asarray, [xtrain, xtest, ytrain, ytest])


def normalize(y, params=None):
    """
    params : (mu, std)
    """

    if params is None:
        mu = np.mean(y)
        std = np.std(y)
    else:
        mu, std = params

    return mu, std, (y - mu) / std


def plot_data(train, test):
    xtrain, ytrain = train
    xtest, ytest = test
    # hu_train, _ = zip(*xtrain)
    # hu_test, _ = zip(*xtest)
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.scatter(ytrain[:,0], xtrain[:,0], s=8);
    plt.scatter(ytest[:,0], xtest[:,0], s=8);
    plt.xlabel("x"); plt.ylabel("y")
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.scatter(ytrain[:,1], xtrain[:,1], s=8);
    plt.scatter(ytest[:,1], xtest[:,1], s=8);
    plt.xlabel("x"); plt.ylabel("y")


def plot_loss(train_loss, test_loss):
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.xlabel("epoch");
    plt.ylabel("loss")
    plt.yscale('log')
    plt.legend()



if __name__ == '__main__':
    """
    Y = [
        [I1],
        [I10],
        [I1],
        [I1],
    ]
    X = [
        [X1, kTe1],
        [X10, kTe3],
        [X1, kTe],
        [X1, kTe],
        [X1, kTe],
    ]
    """

    # xtrain, xtest, ytrain, ytest = get_data()
