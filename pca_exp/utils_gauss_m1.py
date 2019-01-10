import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import load_digits
from tqdm import tqdm


def gen_fake_data(b, e, n, sig1, mu1, sig2, mu2, sig3, mu3):
    x = np.linspace(b, e, n) 
    g1 =  1./(sig1*np.sqrt(2.*np.pi)) * np.exp(-(x - mu1)**2./(2.*sig1**2.))
    g2 =  1./(sig2*np.sqrt(2.*np.pi)) * np.exp(-(x - mu2)**2./(2.*sig2**2.))
    g3 =  1./(sig3*np.sqrt(2.*np.pi)) * np.exp(-(x - mu3)**2./(2.*sig3**2.))
    GT = g1 + g2 + g3     
    y = GT * (1. + 0.01 * np.random.randn(n))
    #y = GT * (1. + 0.01 * torch.randn(n))
    return x, y


def gauss_data():
    hu = np.linspace(1, 10, 200) 
    #num_of_gauss = np.random.randint(1, 2)
    num_of_gauss = 10000
    #I = 0.2*(max(hu) - hu)

    X = []
    Y = []
    for gauss in range(num_of_gauss):
        mu = np.random.choice(np.linspace(2, 8, 10))
        std = np.random.choice(np.linspace(.01, 0.3, 10))
        #amp = (10. - hu) * np.random.random(1); amp;
        amp = np.random.random(1); amp;
        #amp = 1.0
        #I = I + amp * 1./(std*np.sqrt(2.*np.pi)) * np.exp(-(hu - mu)**2./(2.*std**2.))
        I = amp * 1./(std*np.sqrt(2.*np.pi)) * np.exp(-(hu - mu)**2./(2.*std**2.))
        Y.append([mu,std,amp])
        #X.append([I, hu]) 
        X.append(I) 

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y)
    
    #return map(np.asarray, [X,Y])
    # return X,Y
    #return map(np.asarray, [hu, xtrain, xtest, ytrain, ytest])
    return hu, xtrain, ytrain, xtest, ytest


def gen_simple_data():
    x = np.linspace(0, 10, 200) 

    num_of_gauss = np.random.choice(range(1, 5))

    for gauss in range(num_of_gauss):
        mu = gauss
        std = 0.1
        amp = 1 # np.random.random(1)
        y = amp * 1./(std*np.sqrt(2.*np.pi)) * np.exp(-(x - mu)**2./(2.*std**2.))

    plt.figure()
    plt.title('data')
    plt.plot(x,y)

    return x, y 

"""

THIS IS WHAT THE DATA LOOKS LIKE:


In [13]: Xtrain.shape
Out[13]: torch.Size([7500, 8])

In [14]: Ytrain.shape
Out[14]: torch.Size([7500, 3])


Note 8 values of Xtrain in each instance, since 8 PCA components for this run
[12]: Xtrain[::1000]
Out[12]:
tensor([[-2.640e+04, -5.189e+02,  9.109e+01, -4.528e+02,  5.368e+01,  9.097e+01,
         -1.308e+02,  4.527e+01],
        [-1.104e+04, -9.481e+01,  7.756e+01,  5.400e+02, -1.010e+02, -6.432e+01,
          1.786e+01,  1.699e+02],
        [-2.690e+03,  5.593e+02, -5.872e+02, -7.517e+01,  1.277e+02,  1.519e+02,
          5.317e+01, -8.686e+01],
        [ 2.868e+04, -5.401e+02,  4.574e+02,  2.757e+02,  3.777e+02,  1.510e+02,
          1.058e+02,  1.655e+02],
        [ 3.042e+03,  6.280e+02,  5.498e+02, -1.647e+02, -3.081e+00,  4.038e+01,
          9.356e+00, -3.788e+01],
        [ 2.265e+04, -4.251e+02, -1.332e+02,  3.049e+01,  7.008e+01, -6.582e+00,
         -2.983e+01, -9.474e+01],
        [-1.268e+04, -6.592e+01,  3.968e+01,  4.839e+02, -7.326e+01, -1.164e+02,
         -8.661e+01,  1.844e+02],
        [ 1.006e+04,  1.665e+02,  1.416e+02, -5.504e+01, -2.343e+02, -1.710e+02,
          3.570e+01, -1.927e+02]])
In [7]: Ytrain[::100,0]
Out[7]:
tensor([3.333, 8.000, 6.667, 5.333, 5.333, 5.333, 2.667, 4.000, 7.333, 2.000,
        8.000, 2.667, 8.000, 4.667, 6.667, 2.000, 6.667, 2.667, 5.333, 5.333,
        4.667, 4.000, 6.000, 7.333, 2.000, 2.667, 8.000, 7.333, 2.000, 2.000,
        7.333, 5.333, 5.333, 7.333, 4.667, 7.333, 7.333, 5.333, 5.333, 6.000,
        3.333, 3.333, 6.000, 6.667, 4.000, 6.667, 5.333, 4.667, 6.667, 2.000,
        2.000, 4.000, 7.333, 2.667, 5.333, 3.333, 6.000, 3.333, 2.667, 8.000,
        5.333, 7.333, 2.000, 6.667, 2.667, 6.667, 3.333, 4.000, 2.667, 2.000,
        2.667, 4.667, 6.667, 2.000, 2.667])

In [8]: Ytrain[::100,1]
Out[8]:
tensor([0.176, 0.176, 0.238, 0.020, 0.207, 0.207, 0.144, 0.020, 0.269, 0.082,
        0.207, 0.020, 0.238, 0.176, 0.269, 0.207, 0.051, 0.269, 0.238, 0.113,
        0.269, 0.238, 0.238, 0.113, 0.020, 0.082, 0.113, 0.207, 0.082, 0.144,
        0.051, 0.144, 0.300, 0.238, 0.269, 0.020, 0.176, 0.113, 0.269, 0.300,
        0.207, 0.207, 0.082, 0.300, 0.113, 0.269, 0.113, 0.269, 0.020, 0.300,
        0.269, 0.300, 0.176, 0.300, 0.207, 0.207, 0.020, 0.113, 0.144, 0.207,
        0.207, 0.082, 0.207, 0.207, 0.144, 0.082, 0.269, 0.051, 0.144, 0.176,
        0.207, 0.176, 0.082, 0.144, 0.113])

In [9]: Ytrain[::100,2]
Out[9]:
tensor([2.018, 0.838, 1.999, 1.148, 0.507, 3.211, 3.838, 2.086, 1.492, 6.791,
        0.785, 0.711, 1.874, 1.992, 2.267, 5.589, 3.108, 6.188, 1.050, 3.675,
        0.406, 5.197, 2.078, 0.646, 5.103, 4.015, 1.821, 2.160, 0.681, 2.744,
        0.644, 2.106, 1.337, 0.426, 3.337, 1.053, 1.214, 0.294, 4.334, 1.475,
        5.819, 3.478, 0.751, 2.694, 2.190, 3.222, 1.246, 3.483, 1.936, 7.525,
        2.831, 2.917, 1.992, 5.938, 3.326, 2.992, 1.518, 6.244, 4.404, 0.375,
        4.359, 0.452, 6.213, 1.880, 1.934, 0.093, 1.806, 5.241, 0.907, 6.259,
        5.400, 5.145, 1.077, 7.268, 1.340])

"""
