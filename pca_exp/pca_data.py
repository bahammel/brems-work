import numpy as np
import matplotlib.pyplot as plt
import utils_gauss_m1
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from tqdm import tqdm

def pca_data():
    plt.close('all')

    hu, xtrain, ytrain, xtest, ytest,  = utils_gauss_m1.gauss_data()

    scaler = StandardScaler()
    # print(scaler.fit(np.array(X)))
    # Y_std = scaler.fit_transform(np.array(Y))   

    # pca = PCA(n_components=20) # 0.9)
    pca = PCA(0.95)
    pca.fit(xtrain)

    print(f"Data decomposed into {pca.n_components_} components")

    evecs = pca.components_[pca.explained_variance_.argsort()][::-1]
    evals = pca.explained_variance_[pca.explained_variance_.argsort()][::-1]

    plt.figure()
    [plt.plot(vec) for vec in evecs]
    plt.title("eigen vectors")

    fig, axes = plt.subplots(pca.n_components_, 1, figsize=(6, 10))
    plt.title("individual eigen vectors")
    for i, ax in enumerate(axes.flat):
        ax.plot(evecs[i])

    xtrain_hat = pca.transform([xtrain[0]])

    plt.figure()
    plt.plot(xtrain[0], label='data')
    plt.plot(pca.mean_ + np.sum(xtrain_hat.T * pca.components_, axis=0), label='manual reconstruction')   
    plt.plot(pca.inverse_transform(xtrain_hat)[0], linestyle='dashed',label='pca reconstruction')
    plt.legend()
    plt.title("reconstructions")

    return hu, xtrain, ytrain, xtest, ytest, pca.components, pca.mean

