import numpy as np
import matplotlib.pyplot as plt
import utils_3
import torch
from mpl_toolkits.mplot3d import Axes3D

plt.ion()
plt.close('all')
device = torch.device('cuda') # Uncomment this to run on GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.set_default_tensor_type('torch.FloatTensor')

BATCH_SZ, D_in_1, H, D_out = 32, 2, 1600, 2
EPOCHS = 1000000


if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(2),
        torch.nn.Linear(D_in_1, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, D_out),
    )
    model.cuda()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-6, patience=200, factor=.2
    )

    train_losses = []
    test_losses = []

    xtrain, xtest, ytrain, ytest = utils_3.get_data_3()
    utils_3.plot_data((xtrain, ytrain), (xtest, ytest))

    xtrain = torch.Tensor(xtrain)
    ytrain = torch.Tensor(ytrain)
    xtest = torch.Tensor(xtest)
    ytest = torch.Tensor(ytest)

    for epoch in range(EPOCHS):

        epoch_loss = []

        for batch_idx in range(len(xtrain) // BATCH_SZ):
            x_batch = xtrain[batch_idx*BATCH_SZ:(batch_idx+1)*BATCH_SZ]
            y_batch = ytrain[batch_idx*BATCH_SZ:(batch_idx+1)*BATCH_SZ]

            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)
            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)

        # y_pred = model(xtest)

        # test_loss = loss_fn(y_pred, ytest)
        # test_losses.append(test_loss)
        avg_loss = np.mean(epoch_loss)
        train_losses.append(avg_loss)

        if epoch % (EPOCHS/100) == 0:
            print('-'*100)
            print(f"epoch: {epoch}, train loss: {avg_loss}")
            #print(f"epoch: {epoch}, test loss: {test_loss}")


    utils_3.plot_loss(train_losses, test_losses)

    I_ = model(xtest)

    fig = plt.figure(dpi=100, figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xtest[:,0], ytest[:,0], xtest[:,1], c='r', marker='o')
    ax.scatter(xtest[:,0], I_[:,0].detach().squeeze(-1), xtest[:,1], c='m')
    ax.set_xlabel('hu (xtest[:,0])')
    ax.set_ylabel('kTe (ytest[:,0])')
    ax.set_zlabel('Int (xtest[:,1])')

    fig = plt.figure(dpi=100, figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xtest[:,0], ytest[:,0], xtest[:,1], c='r', marker='o')
    ax.scatter(xtest[:,0], I_[:,1].detach().squeeze(-1), xtest[:,1], c='m')
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
