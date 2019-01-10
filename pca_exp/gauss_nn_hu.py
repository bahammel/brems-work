import numpy as np
import matplotlib.pyplot as plt
import pca_data_hu
import torch
import torch.nn as nn
from logger import Logger
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
from tqdm import tqdm


torch.set_printoptions(precision=3) #this doesn't seem to do anything


plt.ion()
plt.close('all')

USE_GPU = torch.cuda.is_available()

device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)

hu, xtrain, ytrain, xtest, ytest, xtrain_pca, xtest_pca, PC = pca_data_hu.pca_data()
print(f"Input Dimension D_in_1 will be: {PC}")

BATCH_SZ, D_in_1, L, M, Q, R, D_out = 2048, PC, 1000, 100, 50, 10, 3
EPOCHS = 20_000


if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(D_in_1),
        torch.nn.Linear(D_in_1, L),
        #torch.nn.Sigmoid(),
        #torch.nn.LeakyReLU(),
        torch.nn.Tanh(),
        torch.nn.Linear(L, M),
        torch.nn.Tanh(),
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(M, Q),
        torch.nn.Tanh(),
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(Q, R),
        torch.nn.Tanh(),
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(R, D_out),
    )

    if USE_GPU:
        model.cuda()

    loss_fn = torch.nn.L1Loss()
    #loss_fn = torch.nn.MSELoss()
    best_loss = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-10, patience=500, factor=0.5, verbose=True,
    )

    experiment_id = datetime.now().isoformat()
    print('Logging experiment as: ', experiment_id)

    logger = Logger(f'/hdd/bahammel/tensorboard/{experiment_id}')

    #utils_4mod.plot_data((xtrain, ytrain), (xtest, ytest))

    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:

        # xtrain, ytrain = shuffle(xtrain, ytrain)

        Xtrain = torch.Tensor(xtrain_pca)
        Ytrain = torch.Tensor(ytrain)
        Xtest = torch.Tensor(xtest_pca)
        Ytest = torch.Tensor(ytest)

        model.train()  # Tell pytorch the model is about to train

        train_losses = []
        for batch_idx in range(len(xtrain) // BATCH_SZ):
            x_batch = Xtrain[batch_idx*BATCH_SZ:(batch_idx+1)*BATCH_SZ]
            y_batch = Ytrain[batch_idx*BATCH_SZ:(batch_idx+1)*BATCH_SZ]

            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        model.eval()  # tell pytorch the model is in test mode

        y_pred = model(Xtest)
        test_loss = loss_fn(y_pred, Ytest)

        lr_scheduler.step(test_loss)

        test_loss = test_loss.item()

        pbar.set_description(
            f"mean train loss: {train_loss:.4f}, mean test loss: {test_loss:.4f}"
        )

        logger.scalar_summary('lr', optimizer.param_groups[0]['lr'], epoch)
        logger.scalar_summary('train loss', train_loss, epoch)
        logger.scalar_summary('test loss', test_loss, epoch)

        # 2. Log values and gradients of the parameters (histogram summary)
        # curious to see if this works!
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            # logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)

        if best_loss < test_loss:
            torch.save(model, f'/hdd/bahammel/checkpoint/{experiment_id}')
            best_loss = test_loss


    I_ = model(Xtest)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.scatter(Ytrain[::100,0], Xtrain.mean(1)[::100], c='g')
    plt.scatter(Ytest[::100,0],Xtest.mean(1)[::100],  c='r')
    plt.scatter(I_[::100,0].detach().squeeze(-1), Xtest.mean(1)[::100],  c='m')
    plt.ylabel('Int (Xtest.mean(1))')
    plt.xlabel('mu (Ytest[:,0])') 
    
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.scatter(Ytrain[::100,1], Xtrain.mean(1)[::100], c='g')
    plt.scatter(Ytest[::100,1],Xtest.mean(1)[::100],  c='r')
    plt.scatter(I_[::100,1].detach().squeeze(-1), Xtest.mean(1)[::100],  c='m')
    plt.ylabel('Int (Xtest.mean(1)')
    plt.xlabel('std (Ytest[:,1])')
 
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.scatter(Ytrain[::100,2], Xtrain.mean(1)[::100], c='g')
    plt.scatter(Ytest[::100,2],Xtest.mean(1)[::100],  c='r')
    plt.scatter(I_[::100,2].detach().squeeze(-1), Xtest.mean(1)[::100],  c='m')
    plt.ylabel('Int (Xtest.mean(1)')
    plt.xlabel('amp (Ytest[:,2])')
    

    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Ytest[::10,0], Ytest[::10,1], Ytest[::10,2], c='r', marker='o')
    ax.scatter(Ytrain[::10,0], Ytrain[::10,1], Ytrain[::10,2], c='g', marker='o')
    ax.scatter(I_[::10,0].detach().squeeze(-1), I_[::10,1].detach().squeeze(-1), I_[::10,2].detach().squeeze(-1), c='m', marker='o') 
    ax.set_xlabel('mu (Ytest[:,0])')
    ax.set_ylabel('std (Ytest[:,1])')
    ax.set_zlabel('amp (Ytest[:,2])')
