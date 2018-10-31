import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import torch

"""
A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x by minimizing squared Euclidean distance.
This implementation computes the forward pass using operations on PyTorch
Tensors, and uses PyTorch autograd to compute gradients.
When we create a PyTorch Tensor with requires_grad=True, then operations
involving that Tensor will not just compute values; they will also build up
a computational graph in the background, allowing us to easily backpropagate
through the graph to compute gradients of some Tensors with respect to a
downstream loss. Concretely if x is a Tensor with x.requires_grad == True then
after backpropagation x.grad will be another Tensor holding the gradient of x
with respect to some scalar value.
"""
plt.close('all')
losses = []

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 100, 100, 100
EPOCHS = 20_000

# Create random Tensors to hold input and outputs
# x = torch.randn(N, D_in, device=device)
# y = torch.randn(N, D_out, device=device)
# x = torch.randn(1, device=device)
# y = torch.randn(1, device=device)

# Create random Tensors for weights; setting requires_grad=True means that we
# want to compute gradients for these Tensors during the backward pass.
w1 = torch.randn((D_in, D_in), device=device, requires_grad=True)
w2 = torch.randn((D_out, D_out), device=device, requires_grad=True)
# w1 = torch.randn(D_in, device=device, requires_grad=True)
# w2 = torch.randn(D_in, device=device, requires_grad=True)

# Here we generate some fake data
def lin(u1,u2,x): return u1*x+u2

Z = 1.
ne = 1.e20
ni = ne
kTe = 1.
n = D_in

def brems(n, ne, kTe, Z):
    # x = torch.randn(1, n, device=device) 
    # y = lin(u1,u2,x) + 0.01 * torch.randn(1, n, device=device)
    # x = torch.linspace(1, 5, n)
    x = torch.linspace(1, 5, n).reshape(1, n)
    # y = 1.e-5 * 5.34e-39 * Z**2. * ne**2.* (1.6e-12 * kTe)**-0.5 * np.exp(-x/kTe)
    y = 1.e-5 * 5.34e-39 * Z**2. * ne**2.* (1.6e-12 * kTe)**-0.5 * np.exp(-x/kTe)  + 1.0 * torch.randn(1, n, device=device)
    return x, y

def normalize(x, y):
    y_norm = (y-y.min())/(y.max()-y.min())
    x_norm = (x-x.min())/(x.max()-x.min())
    return x_norm, y_norm

def sigmoid(z):
    return torch.sigmoid(z)

x_true, y_true = brems(n, ne, kTe, Z)


# def mse(y_hat, y): return ((y_hat - y) ** 2).mean()
# def mse_loss(w1, w2, x, y): return mse(lin(w1,w2,x), y)

x, y = normalize(x_true, y_true)

plt.scatter(x, y, s=8);
plt.xlabel("x")
plt.ylabel("y")

learning_rate = 1e-5
for t in range(EPOCHS):
    # Forward pass: compute predicted y using operations on Tensors. Since w1 and
    # w2 have requires_grad=True, operations involving these Tensors will cause
    # PyTorch to build a computational graph, allowing automatic computation of
    # gradients. Since we are no longer implementing the backward pass by hand we
    # don't need to keep references to intermediate values.
    y_pred = sigmoid(x.mm(w1)).mm(w2)

    # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
    # is a Python number giving its value.

    # regulizer = 1e-1*(w2.pow(2).sum() + w1.pow(2).sum())
    loss = (y_pred - y).pow(2).sum() # + regulizer
    # print(t, loss.item())
    
    # loss = mse_loss(w1,w2,x,y)
    # print(t, loss.item())
    losses.append(loss.item())
    
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()
    
    # Update weights using gradient descent. For this step we just want to mutate
    # the values of w1 and w2 in-place; we don't want to build up a computational
    # graph for the update steps, so we use the torch.no_grad() context manager
    # to prevent PyTorch from building a computational graph for the updates
    if torch.isnan(w1).any() or torch.isnan(w2).any() or torch.isnan(loss):
        print("nan value in tensor")
        assert False
  
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()

    if t % (EPOCHS/100) == 0:
        print(f"epoch={t}, loss={loss}")


plt.plot(x.data.numpy(), y_pred.data.numpy(), 'go')
# plt.ylim(0.1,175.)
plt.figure()
plt.plot(losses)
plt.yscale('log')
