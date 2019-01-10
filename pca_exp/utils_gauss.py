import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from tqdm import tqdm

#def get_mnist(): X, Y = load_digits(return_X_y=True)
#    return X

def gen_fake_data(b, e, n, sig1, mu1, sig2, mu2, sig3, mu3):
    x = np.linspace(b, e, n) 
    g1 =  1./(sig1*np.sqrt(2.*np.pi)) * np.exp(-(x - mu1)**2./(2.*sig1**2.))
    g2 =  1./(sig2*np.sqrt(2.*np.pi)) * np.exp(-(x - mu2)**2./(2.*sig2**2.))
    g3 =  1./(sig3*np.sqrt(2.*np.pi)) * np.exp(-(x - mu3)**2./(2.*sig3**2.))
    GT = g1 + g2 + g3     
    y = GT * (1. + 0.01 * np.random.randn(n))
    #y = GT * (1. + 0.01 * torch.randn(n))
    return x, y


def custom_pca(X):
    X_std = StandardScaler().fit_transform(X)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the eigenvalue, eigenvector pair from high to low
    eig_pairs.sort(key = lambda x: x[0], reverse= True)

    eig_vecs = [vec[1] for vec in eig_pairs]
    return eig_vecs


def gen_fake_data2():
    x = np.linspace(0, 10, 200) 

    num_of_gauss = np.random.randint(1, 2)

    y = 0.2*(max(x) - x)
    for gauss in range(num_of_gauss):
        mu = np.random.choice(np.linspace(2, 8, 4))
        std = np.random.choice(np.linspace(.02, .3, 5))
        amp = (10. - mu) * np.random.random(1); amp;
        y = y + amp * 1./(std*np.sqrt(2.*np.pi)) * np.exp(-(x - mu)**2./(2.*std**2.))

    y = y + 0.01 * np.random.randn(len(y))
    plt.plot(x,y)

    return x, y 


def gen_simple_data():
    x = np.linspace(0, 10, 200) 

    num_of_gauss = np.random.choice(range(1, 5))

    for gauss in range(num_of_gauss):
        mu = gauss
        std = .1
        amp = 1 # np.random.random(1)
        y = amp * 1./(std*np.sqrt(2.*np.pi)) * np.exp(-(x - mu)**2./(2.*std**2.))

    plt.figure()
    plt.title('data')
    plt.plot(x,y)

    return x, y 


if __name__ == '__main__':
    plt.close('all')

    X = []
    Y = []
    for _ in tqdm(range(1000)):
        x, y = gen_fake_data2()
        X.append(x)
        Y.append(y)

    scaler = StandardScaler()
    # print(scaler.fit(np.array(X)))
    # Y_std = scaler.fit_transform(np.array(Y))   

    # pca = PCA(n_components=20) # 0.9)
    pca = PCA(0.999)
    pca.fit(Y)

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

    Y_hat = pca.transform([Y[0]])

    plt.figure()
    plt.plot(Y[0], label='data')
    plt.plot(pca.mean_ + np.sum(Y_hat.T * pca.components_, axis=0), label='manual reconstruction')
    plt.plot(pca.inverse_transform(Y_hat)[0], label='pca reconstruction')
    plt.legend()
    plt.title("reconstructions")

