import numpy as np
import matplotlib.pyplot as plt
from utils import get_data, normalize, brems
plt.ion()
import torch

Z = 1.
ne = 1.e20
ni = ne

plt.close('all')
losses = []

# device = torch.device('cuda') # Uncomment this to run on GPU
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type('torch.FloatTensor')

# D_in is input dimension; H is hidden dimension; D_out is output dimension.
BATCH_SZ, D_in, H, D_out = 128, 2, 1600, 1

EPOCHS = 500

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, D_out),
)
# model.cuda()

loss_fn = torch.nn.MSELoss()

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []

fig = plt.figure(dpi=100, figsize=(5, 4))

xtrain, xtest, ytrain, ytest = get_data()

*params, ytest = normalize(ytest)
*_, ytrain = normalize(ytrain, params=params)

hu_train, _ = zip(*xtrain)
hu_test, _ = zip(*xtest)
plt.scatter(hu_train, ytrain, s=8);
plt.scatter(hu_test, ytest, s=8);
plt.xlabel("x"); plt.ylabel("y")

xtrain = torch.Tensor(xtrain)
ytrain = torch.Tensor(ytrain)
xtest = torch.Tensor(xtest)
ytest = torch.Tensor(ytest)

BATCH_NUM = len(xtrain) // BATCH_SZ

for epoch in range(EPOCHS):

    epoch_loss = []
    
    for batch_idx in range(BATCH_NUM):
        x_batch = xtrain[BATCH_NUM*BATCH_SZ:(BATCH_NUM+1)*BATCH_SZ]
        y_batch = ytrain[BATCH_NUM*BATCH_SZ:(BATCH_NUM+1)*BATCH_SZ]

        y_pred = model(x_batch)

        loss = loss_fn(y_pred, y_batch)
        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    y_pred = model(xtest)
    test_loss = loss_fn(y_pred, ytest)
    test_losses.append(test_loss)
    avg_loss = np.mean(epoch_loss)
    train_losses.append(avg_loss)

    if epoch % (EPOCHS/100) == 0:
        print('-'*100)
        print(f"epoch: {epoch}, train loss: {avg_loss}")
        print(f"epoch: {epoch}, test loss: {test_loss}")


fig = plt.figure(dpi=100, figsize=(5, 4))
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.xlabel("epoch");
plt.ylabel("loss")
plt.yscale('log')
plt.legend()


fig = plt.figure('test', dpi=100, figsize=(5, 4))
I_ = model(xtest)
hu, _ = list(zip(*xtest))
plt.plot(hu, ytest.cpu().data.numpy().reshape(-1), 'o')
plt.plot(hu, I_.cpu().data.numpy().reshape(-1), 'o')

fig = plt.figure('train', dpi=100, figsize=(5, 4))
I_ = model(xtrain)
hu, _ = list(zip(*xtrain))
plt.plot(hu, ytrain.cpu().data.numpy().reshape(-1), 'o')
plt.plot(hu, I_.cpu().data.numpy().reshape(-1), 'o')
