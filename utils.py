import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
>>>>>>> Try batched brems learning
from sklearn.model_selection import train_test_split
import torch


Z = 1.
ne = 1.e20
ni = ne


def brems(ne, kTe, Z, x):
    y = 1.e-5 * 5.34e-39 * Z**2. * ne**2.* (1.6e-12 * kTe)**-0.5 * np.exp(-x/kTe)    
    return y


def get_data():
<<<<<<< HEAD
    kTe = np.linspace(1000, 5000, 5)
=======
    kTe = np.linspace(1000, 5000, 2)
>>>>>>> Try batched brems learning
    x = np.linspace(1000, 5100, 1000)

    X = []
    Y = []

    for t in kTe:
        for i in x:
            X.append([i, t])
            Y.append([brems(ne, t, Z, i)])

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

<<<<<<< HEAD
    return map(np.asarray, [xtrain, xtest, ytrain, ytest])
=======
    return xtrain, xtest, ytrain, ytest
>>>>>>> Try batched brems learning


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


<<<<<<< HEAD
def plot_data(train, test):
    fig = plt.figure(dpi=100, figsize=(5, 4))
    xtrain, ytrain = train
    xtest, ytest = test
    hu_train, _ = zip(*xtrain)
    hu_test, _ = zip(*xtest)
    plt.scatter(hu_train, ytrain, s=8);
    plt.scatter(hu_test, ytest, s=8);
    plt.xlabel("x"); plt.ylabel("y")


def plot_loss(train_loss, test_loss):
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.xlabel("epoch");
    plt.ylabel("loss")
    plt.yscale('log')
    plt.legend()



=======
>>>>>>> Try batched brems learning
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

    xtrain, xtest, ytrain, ytest = get_data()
