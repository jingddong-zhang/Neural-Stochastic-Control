import numpy as np
from cvxopt import solvers,matrix
import matplotlib.pyplot as plt
import torch


def harmonic(n,dt):
    x0 = np.array([2.0,2.0])
    X = np.zeros([n,2])
    X[0,:]=x0
    z = np.random.normal(0, 1, n)
    for i in range(n-1):
        x1,x2 = X[i,:]
        X[i+1,0] = x1 + (x2-4.45*x1-0.09*x2)*dt
        X[i+1,1] = x2 + (-x1-x2-0.09*x1-3.6*x2)*dt+(-3*x1+2.15*x2)*np.sqrt(dt)*z[i]
    return X

n = 6000
dt = 0.0001
X = np.zeros([10,n,2])
for i in range(10):
    np.random.seed(20*i)
    X[i,:] = harmonic(n,dt)
np.save('lqr.npy',X)

# X = harmonic(n,dt)
# plt.plot(np.arange(len(X)),X[:,0])
# plt.plot(np.arange(len(X)),X[:,1])
# plt.show()
