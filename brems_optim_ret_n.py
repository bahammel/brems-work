import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import torch


plt.close('all')
losses = []

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
n = 1000
N, D_in, H, D_out = 1, n, 10, n
EPOCHS = 20_000

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    # torch.nn.ReLU(),
    torch.nn.Sigmoid(),
    # torch.nn.functional.sigmoid(),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss(reduction='sum')


# Create random Tensors to hold input and outputs
# x = torch.randn(N, D_in, device=device)
# y = torch.randn(N, D_out, device=device)
# x = torch.randn(1, device=device)
# y = torch.randn(1, device=device)



x = torch.linspace(1, 5, n).reshape(1, n)

# Create random Tensors for weights; setting requires_grad=True means that we
# want to compute gradients for these Tensors during the backward pass.
w1 = torch.randn(D_in, D_in, device=device, requires_grad=True)
w2 = torch.randn(D_out, D_out, device=device, requires_grad=True)


# Here we generate some fake data

a = 1.
b = 1.
c = 1.

def lin(a,b,x):
    y = a*x + b
    return x, y

def quad(a,b,c,x):
    a*(x**2.) + b*x + c
    return x, y

Z = 1.
ne = 1.e20
ni = ne
kTe = 1.

def brems(ne, kTe, Z, x):
    # x = torch.randn(1, n, device=device) 
    # y = lin(u1,u2,x) + 0.01 * torch.randn(1, n, device=device)
    # x = torch.linspace(1, 5, n)
    # y = 1.e-5 * 5.34e-39 * Z**2. * ne**2.* (1.6e-12 * kTe)**-0.5 * np.exp(-x/kTe)
    y = 1.e-5 * 5.34e-39 * Z**2. * ne**2.* (1.6e-12 * kTe)**-0.5 * np.exp(-x/kTe)  + 1.0 * torch.randn(1, n, device=device)
    return x, y

def normalize(x, y):
     y_norm = (y-y.min())/(y.max()-y.min() + 1e-6)
     x_norm = (x-x.min())/(x.max()-x.min() + 1e-6)
     return x_norm, y_norm

def sigmoid(z):
     return torch.sigmoid(z)

x_true, y_true = brems(ne, kTe, Z, x)
# x_true, y_true = quad(a, b, c, x)
# x_true, y_true = lin(a, b, x)


x, y = normalize(x_true, y_true)

fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(x, y, s=8);
plt.xlabel("x"); plt.ylabel("y")

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.

learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(30_000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    losses.append(loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

      
plt.plot(x.data.numpy(), y_pred.data.numpy() + 0.1, 'go')
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)

fig = plt.figure(dpi=100, figsize=(5, 4))
# plt.figure()
plt.plot(losses)
plt.xlabel("epoch"); plt.ylabel("loss")
plt.yscale('log')
