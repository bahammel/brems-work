import numpy as np
import matplotlib.pyplot as plt
import utils_4mod
import torch
import torch.nn as nn
from datetime import datetime
from logger import Logger
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

BATCH_SZ, D_in_1, H, L, Q, R, D_out = 2048, 100, 1000, 500, 50, 10, 2
#BATCH_SZ, D_in_1, H, L, Q,  D_out = 32, 1000, 500, 50, 10, 2
#BATCH_SZ, D_in_1, H, L, Q,  D_out = 32, 100, 50, 50, 10, 2
EPOCHS = 10_000


if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(D_in_1),
        torch.nn.Linear(D_in_1, H),
        #torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(H, L),
        #torch.nn.Sigmoid(),[1;2B
        torch.nn.Tanh(),
        torch.nn.Linear(L, Q),
        torch.nn.Tanh(),
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(Q, R),
        torch.nn.Tanh(),
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(R, D_out),
    )

    if USE_GPU:
        model.cuda()

    loss_fn = torch.nn.MSELoss()
    best_loss = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-9, patience=100, factor=0.1, verbose=True,
    )

    experiment_id = datetime.now().isoformat()
    print('Logging experiment as: ', experiment_id)

    logger = Logger(f'/hdd/bahammel/tensorboard/{experiment_id}')

    hu, xtrain, xtest, ytrain, ytest = utils_4mod.get_data_2m()
    utils_4mod.plot_data((xtrain, ytrain), (xtest, ytest))

    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:

        # xtrain, ytrain = shuffle(xtrain, ytrain)

        Xtrain = torch.Tensor(xtrain)
        Ytrain = torch.Tensor(ytrain)
        Xtest = torch.Tensor(xtest)
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
    '''
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(Xtest[:,0], Ytest[:,0], Xtest[:,1], c='r', marker='o')
    ax.scatter(Xtest[:,0], I_[:,0].detach().squeeze(-1), Xtest[:,1], c='m')
    ax.scatter(Xtest, Ytest[:,0], hu, c='r', marker='o')
    ax.scatter(Xtest, I_[:,0].detach().squeeze(-1), x, c='m')
    ax.set_xlabel('hu (xtest[:,0])')
    ax.set_ylabel('kTe (ytest[:,0])')
    ax.set_zlabel('Int (xtest[:,1])')
    '''
    plt.scatter(Ytrain[::10,0], Xtrain.mean(1)[::10], c='g')
    plt.scatter(Ytest[::10,0],Xtest.mean(1)[::10],  c='r')
    plt.scatter(I_[::10,0].detach().squeeze(-1), Xtest.mean(1)[::10],  c='m')
    plt.ylabel('Int (Xtest.mean(1))')
    plt.xlabel('kTe (Ytest[:,0])')
 
    
    fig = plt.figure(dpi=100, figsize=(5, 4))
    """
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xtest[:,0], Ytest[:,0], Xtest[:,1], c='r', marker='o')
    ax.scatter(Xtest[:,0], I_[:,1].detach().squeeze(-1), Xtest[:,1], c='m')
    ax.scatter(Xtest, Ytest[:,0], hu, c='r', marker='o')
    ax.scatter(Xtest, I_[:,1].detach().squeeze(-1), x, c='m')
    ax.set_xlabel('hu (xtest[:,0])')
    ax.set_ylabel('ne (ytest[:,0])')
    ax.set_zlabel('Int (xtest[:,1])')
    """

    plt.scatter(Ytrain[::10,1], Xtrain.mean(1)[::10], c='g')
    plt.scatter(Ytest[::10,1],Xtest.mean(1)[::10],  c='r')
    plt.scatter(I_[::10,1].detach().squeeze(-1), Xtest.mean(1)[::10],  c='m')
    plt.ylabel('Int (Xtest.mean(1)')
    plt.xlabel('ne (Ytest[:,1])')
 

    

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
