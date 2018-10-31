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
N, D_in, H, D_out = 1, 1000, 100, 1000

# Create random Tensors to hold input and outputs
# x = torch.randn(N, D_in, device=device)
# y = torch.randn(N, D_out, device=device)
x = torch.randn(1, device=device)
y = torch.randn(1, device=device)

# Create random Tensors for weights; setting requires_grad=True means that we
# want to compute gradients for these Tensors during the backward pass.
w1 = torch.randn(D_in, D_in, device=device, requires_grad=True)
w2 = torch.randn(D_out, D_out, device=device, requires_grad=True)
# w1 = torch.randn(D_in, device=device, requires_grad=True)
# w2 = torch.randn(D_in, device=device, requires_grad=True)

# Here we generate some fake data
def lin(u1,u2,x): return u1*x+u2

def gen_fake_data(n, u1, u2):
    x = torch.randn(1, n, device=device) 
    y = lin(u1,u2,x) + 0.01 * torch.randn(1, n, device=device)
    return x, y

x, y = gen_fake_data(1000, 3., 10)


plt.scatter(x,y, s=8); plt.xlabel("x"); plt.ylabel("y");

def mse(y_hat, y): return ((y_hat - y) ** 2).mean()
def mse_loss(w1, w2, x, y): return mse(lin(w1,w2,x), y)

learning_rate = 1e-7
for t in range(30_000):
  # Forward pass: compute predicted y using operations on Tensors. Since w1 and
  # w2 have requires_grad=True, operations involving these Tensors will cause
  # PyTorch to build a computational graph, allowing automatic computation of
  # gradients. Since we are no longer implementing the backward pass by hand we
  # don't need to keep references to intermediate values.
    
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

  # Compute and print loss. Loss is a Tensor of shape (), and loss.item()
  # is a Python number giving its value.

    loss = (y_pred - y).pow(2).sum()
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
  
    with torch.no_grad():
      w1 -= learning_rate * w1.grad
      w2 -= learning_rate * w2.grad

    # Manually zero the gradients after running the backward pass
      w1.grad.zero_()
      w2.grad.zero_()

      

plt.plot(x.data.numpy(), y_pred.data.numpy(), 'go')
plt.figure()
plt.plot(losses)
