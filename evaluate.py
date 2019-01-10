import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils_4
import torch

"""
%run evaluate.py --model='/hdd/bahammel/checkpoint/2018-11-19T21:50:13.039770'
"""

USE_GPU = torch.cuda.is_available()

device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)

BATCH_SIZE = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = torch.load(args.model)

    _, test = utils_4.get_data_3()
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    X = []
    Y = []
    T = []
    for x_batch, t_batch in test_loader:

        y_batch = model(x_batch)

        X.extend(x_batch.cpu().detach().numpy())
        Y.extend(y_batch.cpu().detach().numpy())
        T.extend(t_batch.cpu().detach().numpy())

    X = np.asarray(X)
    Y = np.asarray(Y)
    T = np.asarray(T)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], Y[:,0], X[:,1], c='r', marker='o')
    ax.scatter(X[:,0], T[:,0], X[:,1], c='m')
    ax.set_xlabel('hu (xtest[:,0])')
    ax.set_ylabel('kTe (ytest[:,0])')
    ax.set_zlabel('Int (xtest[:,1])')

    fig = plt.figure(dpi=100, figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], Y[:,1], X[:,1], c='r', marker='o')
    ax.scatter(X[:,0], T[:,1], X[:,1], c='m')
    ax.set_xlabel('hu (xtest[:,0])')
    ax.set_ylabel('ne (ytest[:,0])')
    ax.set_zlabel('Int (xtest[:,1])')


"""
    fig = plt.figure('train', dpi=100, figsize=(5, 4))
    I_ = model(xtrain)
    hu, _ = list(zip(*xtrain))
    plt.plot(hu, ytrain.cpu().data.numpy().reshape(-1), 'o')
    plt.plot(hu, I_.cpu().data.numpy().reshape(-1), 'o')
    plt.xlabel("hu");
    plt.xlabel("kT");
"""

"""
In [85]: xtest[:,1]
Out[85]:
tensor([347.5146, 417.4397, 363.8300, 410.5401, 491.1190,   4.0209,   0.5442,
        269.5319, 341.7708,   0.2002, 253.1900,   0.8118,  10.9867,  81.1814,
          2.8930,  29.8650,   2.5745,  89.7194,   0.2702,   3.5929,   2.2067,
          4.9112,  49.2390,   2.7519,   4.5372, 163.4793,   0.1214,   2.6617,
          1.9967,  44.5533, 384.0631,  99.1552,   2.9911,   2.9416,   3.3612,
          1.3385, 266.1713, 353.3551, 299.1086,  22.1245, 438.8423])

In [86]: xtest[:,0]
Out[86]:
tensor([2700., 1600., 1300., 1700., 1000., 1200., 3200., 1600., 2800., 4200.,
        4600., 2800., 4800., 2800., 3800., 3800., 4500., 2700., 3900., 2500.,
        1800., 1000., 3300., 4100., 1100., 2100., 4700., 4300., 1900., 3400.,
        2100., 2600., 3600., 3700., 2900., 2300., 4300., 2600., 3600., 4100.,
        1300.])

In [87]: ytest[:,0]
Out[87]:
tensor([6., 6., 1., 6., 1., 1., 1., 1., 6., 1., 6., 1., 1., 1., 6., 1., 6., 1.,
        1., 6., 1., 1., 1., 6., 6., 1., 1., 6., 1., 1., 6., 1., 6., 6., 6., 1.,
        6., 6., 6., 1., 6.])

In [88]: ytest[:,1]
Out[88]:
tensor([10., 10., 10., 10., 10.,  1.,  1., 10., 10.,  1., 10.,  1., 10., 10.,
         1., 10.,  1., 10.,  1.,  1.,  1.,  1., 10.,  1.,  1., 10.,  1.,  1.,
         1., 10., 10., 10.,  1.,  1.,  1.,  1., 10., 10., 10., 10., 10.])
"""
