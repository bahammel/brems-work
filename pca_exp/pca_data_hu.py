import numpy as np
import matplotlib.pyplot as plt
import utils_gauss_hu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from tqdm import tqdm

def pca_data():
    plt.close('all')

    hu, xtrain, ytrain, xtest, ytest  = utils_gauss_hu.gauss_data_2()

    scaler = StandardScaler()
    # print(scaler.fit(np.array(X)))
    #>>>>xtrain_std = scaler.fit_transform(np.array(xtrain))  

    # Fit on training set only.
    scaler.fit(xtrain)
    # Apply transform to both the training set and the test set.
    #xtrain_use = scaler.transform(xtrain)
    #xtest_use = scaler.transform(xtest)
    xtrain_use = xtrain
    xtest_use = xtest

    """
    IT DOES LOOK LIKE STANDARDIZATION DOES SOMETHING!
    >
    In [17]: np.array(xtrain_std)[0,::10]
    Out[17]:
    a>rray([0.83848684, 0.84233682, 0.85513213, 0.84763035, 0.77922881,
           0.84469112, 0.818827  , 0.80342314, 0.84740634, 0.84341608,
           0.82762121, 0.82015794, 0.82115436, 0.79986662, 0.81393572,
           0.81309896, 0.80421902, 0.82792931, 0.82839311, 0.84097237])

    In [18]: np.array(xtrain)[0,::10]
    Out[18]:
    array([2.00000017e+00, 1.90266701e+00, 6.17203942e+00, 6.09976236e+02,
           2.58567715e+04, 6.38488020e+03, 3.27297661e+03, 4.72276215e+03,
           1.69010090e+04, 4.91775466e+03, 2.38223264e+03, 3.21376833e+03,
           1.02925583e+04, 3.21941088e+03, 1.39249956e+03, 1.64576213e+03,
           3.81205426e+03, 9.53301824e+01, 7.31015169e-01, 9.07621730e-02])

    """

    #pca = PCA(n_components=20) 
    pca = PCA(0.99)
    pca.fit(xtrain_use)
    #xtrain_hat = pca.transform([xtrain[0]])
    PC = pca.n_components_ 
    print(f"Data decomposed into {PC} components")

    evecs = pca.components_[pca.explained_variance_.argsort()][::-1]
    evals = pca.explained_variance_[pca.explained_variance_.argsort()][::-1]

    plt.figure()
    [plt.plot(vec) for vec in evecs]
    plt.title("eigen vectors")

    fig, axes = plt.subplots(pca.n_components_, 1, figsize=(6, 10))
    plt.title("individual eigen vectors")
    for i, ax in enumerate(axes.flat):
        ax.plot(evecs[i])

    xtrain_pca = pca.transform(xtrain_use)
    xtest_pca = pca.transform(xtest_use)

    plt.figure()
    plt.plot(xtrain_use[0], label='data')
    #plt.plot(pca.mean_ + np.sum(xtrain_pca[0].T * pca.components_, axis=0), label='manual reconstruction')   A
    plt.plot(pca.inverse_transform(xtrain_pca)[0], linestyle='dashed',label='pca reconstruction')
    #xtrain_pca = scaler.transform(xtrain)
    #xtest_pca = scaler.transform(xtest)
    plt.legend()
    plt.title("reconstructions")

    return hu, xtrain, ytrain, xtest, ytest, xtrain_pca, xtest_pca, PC
