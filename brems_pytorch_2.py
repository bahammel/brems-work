import numpy as np
import matplotlib.pyplot as plt
import utils
import torch

plt.ion()
plt.close('all')
# device = torch.device('cuda') # Uncomment this to run on GPU
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type('torch.FloatTensor')

BATCH_SZ, D_in_1, H, D_out = 32, 2, 200, 1
EPOCHS = 50000


if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in_1, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, D_out),
    )
    # model.cuda()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-6, patience=20, factor=.2
    )

    train_losses = []
    test_losses = []

    xtrain, xtest, ytrain, ytest = utils.get_data_2()
    utils.plot_data((xtrain, ytrain), (xtest, ytest))

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


    utils.plot_loss(train_losses, test_losses)

    fig = plt.figure('test', dpi=100, figsize=(5, 4))
    I_ = model(xtest)
    hu, _ = list(zip(*xtest))
    plt.plot(hu, ytest.cpu().data.numpy().reshape(-1), 'o')
    plt.plot(hu, I_.cpu().data.numpy().reshape(-1), 'o')
    plt.xlabel("hu");
    plt.xlabel("kT");



    fig = plt.figure('train', dpi=100, figsize=(5, 4))
    I_ = model(xtrain)
    hu, _ = list(zip(*xtrain))
    plt.plot(hu, ytrain.cpu().data.numpy().reshape(-1), 'o')
    plt.plot(hu, I_.cpu().data.numpy().reshape(-1), 'o')
    plt.xlabel("hu");
    plt.xlabel("kT");
