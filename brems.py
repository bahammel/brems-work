# %matplotlib inline
import math,sys,os,numpy as np
from numpy.random import random
from matplotlib import pyplot as plt, rcParams, animation, rc
# from __future__ import print_function, division
from ipywidgets import interact, interactive, fixed
from ipywidgets.widgets import *
rc('animation', html='html5')
rcParams['figure.figsize'] = 3, 3
# %precision 4
# np.set_printoptions(precision=4, linewidth=100)

Z = 1
ne = 1e20
ni = ne
kTe = 1.
Te = 1.6e-12 * kTe


# This is a comment

def brems(ne,Te,x): return 1.e-5 * 5.34e-39 * Z**2. * ne**2.* Te**-0.5 * np.exp(-x/kTe)
def lin(a,b,x): return a*x+b


n=100

x = np.linspace(1, 5, n)
y = brems(ne,Te,x)

"""
plt.scatter(x,y)
plt.yscale('symlog')
plt.ylim(10.,1.e6)
"""

def sse(y,y_pred): return ((y-y_pred)**2).sum()
def loss(y,a,b,x): return sse(y, brems(a,b,x))
def avg_loss(y,a,b,x): return np.sqrt(loss(y,a,b,x)/n)


a_guess= 1.
b_guess= 10.

avg_loss(y, a_guess, b_guess, x)

lr=1.e-2

def upd():
    global a_guess, b_guess
    y_pred = lin(a_guess, b_guess, x)
    dydb = 2 * (y_pred - y)
    dyda = x*dydb
    a_guess -= lr*dyda.mean()
    b_guess -= lr*dydb.mean()



fig = plt.figure(dpi=100, figsize=(5, 4))
plt.scatter(x,y); plt.xlabel("x"); plt.ylabel("y");
# plt.yscale('symlog')
plt.ylim(0.1,175.)

line, = plt.plot(x,lin(a_guess,b_guess,x))
# plt.close()

def animate(i):
    line.set_ydata(lin(a_guess,b_guess,x))
    for i in range(100): upd()
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(0, 400), interval=100)
ani
