import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from torch.utils import data as data_utils
import torch

Z = 1.
# ne = 1.
# ni = ne

torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'
)


def brems(ne, kTe, Z, x):
    kTe *= 1.e3
    y = (1.e40 * 1.e-5 * 5.34e-39 * Z**2. * (ne)**2.* (1.6e-12 * kTe)**-0.5 * np.exp(-(1.e3*x)/kTe))**0.25  #Note I am taking the fourth root to compress range of Intensity
    return y


def get_data_2m():
    kTe = np.linspace(1, 6, 500)
    ne = np.linspace(1, 10, 500)
    hu = np.linspace(1, 5, 100)

    X = []
    Y = []

    for t in kTe:
        for k in ne:    
            Y.append([t,k])
            X.append(brems(k, t, Z, hu))

    #X = brems(hu, ne, kTe, Z)

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y)
    
    return map(np.asarray, [hu, xtrain, xtest, ytrain, ytest])


"""
THIS IS WHAT THE DATA LOOKS LIKE:

In [35]: Xtrain
Out[35]:
tensor([[4.888, 4.885, 4.883,  ..., 3.137, 3.135, 3.134],
        [2.642, 2.642, 2.641,  ..., 2.237, 2.237, 2.236],
        [1.522, 1.521, 1.521,  ..., 1.144, 1.144, 1.144],
        ...,
        [2.691, 2.690, 2.690,  ..., 2.181, 2.180, 2.180],
        [4.708, 4.703, 4.698,  ..., 1.735, 1.734, 1.732],
        [2.786, 2.785, 2.784,  ..., 1.788, 1.787, 1.787]])

In [36]: Xtrain.shape
Out[36]: torch.Size([18, 1001])

In [40]: Xtrain[0,:]
Out[40]: tensor([4.888, 4.885, 4.883,  ..., 3.137, 3.135, 3.134])

In [42]: Xtrain[0,:].mean()
Out[42]: tensor(3.946)

In [1]: Xtrain.mean(1)  THIS AVERAGES OVER THE INTENSITIES FOR ALL hu FOR EACH OF THE 18 ne AND kTe
Out[1]:
tensor([2.976, 3.946, 2.434, 1.350, 3.104, 1.346, 2.926, 3.166, 1.324, 4.269,
        1.248, 0.941, 3.747, 1.697, 2.620, 3.685, 3.758, 2.426])

In [2]: Xtrain.mean(0) THIS AVERAGES OVER INTENSITIES FOR ALL ne and kTe FOR EACH OF THE 1001 hu
Out[2]: tensor([3.167, 3.166, 3.164,  ..., 2.155, 2.155, 2.154])

In [34]: Ytrain
Out[34]:
tensor([[ 2.250, 10.000],
        [ 6.000,  3.250],
        [ 3.500,  1.000],
        [ 2.250,  5.500],
        [ 3.500, 10.000],
        [ 4.750,  1.000],
        [ 2.250,  7.750],
        [ 1.000,  3.250],
        [ 6.000,  1.000],
        [ 3.500,  7.750],
        [ 1.000,  7.750],
        [ 3.500,  3.250],
        [ 6.000, 10.000],
        [ 1.000,  5.500],
        [ 4.750, 10.000],
        [ 4.750,  3.250],
        [ 1.000, 10.000],
        [ 2.250,  3.250]])
In [37]: Ytrain.shape
Out[37]: torch.Size([18, 2])
"""


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
    #plt.scatter(ytrain[:,1], xtrain[:,1], s=8);
    #plt.scatter(ytest[:,1], xtest[:,1], s=8);
    plt.scatter(ytrain[:,1], xtrain[:,0], s=8);
    plt.scatter(ytest[:,1], xtest[:,0], s=8);
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
