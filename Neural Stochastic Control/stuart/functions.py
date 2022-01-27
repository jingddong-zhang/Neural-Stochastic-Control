import torch 
import numpy as np
import timeit 
import matplotlib.pyplot as plt




'''
x = rho_1,rho_2,rho_n, w1,w2,wn-1
'''

#Transform \Tilde{\theta} to \theta
def theta(W):
    W = torch.cat([W,torch.tensor([1.0])],0)
    T = torch.eye(len(W))
    for i in range(len(T)):
        for k in range(len(T)):
            if k>i:
                T[i,k]=1.0
    W = W.unsqueeze(1)
    ang = torch.mm(T,W)
    return ang[:,0]

#Transform \theta to \Tilde{\theta}
def diff_theta(W):
    T = torch.eye(len(W))
    for i in range(len(W)):
        for j in range(len(W)):
            T[i,j] = W[j] - W[i]
    return T

#Equation for \Tilde{\rho},\Tilde{\theta}
def f_value(x,L):
    c1 = -1.8
    c2 = 4
    sigma = 0.01
    k = int((len(x)+1)/2)
    R = x[:k]+1.0
    W = x[k:]
    diff_ang = diff_theta(theta(W))
    f_R = torch.zeros_like(R)
    f_W = torch.zeros_like(W)
    for j in range(len(R)):
        f_R[j] = R[j]-R[j]**3-sigma*torch.sum(L[j,:]*R*(torch.cos(diff_ang[j,:])-c1*torch.sin(diff_ang[j,:])))
    for j in range(len(W)):
        f_W[j] = -c2*(R[j]**2-R[j+1]**2)-sigma*(torch.sum(L[j,:]*R*(c1*torch.cos(diff_ang[j,:])+torch.sin(diff_ang[j,:])))/R[j]\
            -torch.sum(L[j+1,:]*R*(c1*torch.cos(diff_ang[j+1,:])+torch.sin(diff_ang[j+1,:])))/R[j+1])
    return torch.cat([f_R,f_W],0)

#Equation for \rho, \theta
def original_f_value(x,L):
    c1 = -1.8
    c2 = 4
    sigma = 0.01
    k = int(len(x)/2)
    R = x[:k]
    W = x[k:]
    diff_ang = diff_theta(W)
    f_R = torch.zeros_like(R)
    f_W = torch.zeros_like(W)
    for j in range(len(R)):
        f_R[j] = R[j]-R[j]**3-sigma*torch.sum(L[j,:]*R*(torch.cos(diff_ang[j,:])-c1*torch.sin(diff_ang[j,:])))
        f_W[j] = -c2*(R[j]**2)-sigma*(torch.sum(L[j,:]*R*(c1*torch.cos(diff_ang[j,:])+torch.sin(diff_ang[j,:])))/R[j])
    return torch.cat([f_R,f_W],0)

# Transform polar coordinate to euclidean coordinate
def transform(n,X):
    Y = torch.zeros_like(X)
    for i in range(n):
        Y[:,i] = X[:,i]*torch.cos(X[:,i+n])
        Y[:,i+n] = X[:,i]*torch.sin(X[:,i+n])
    return Y

#Generate control data
def generate():
    N = 5000             
    n = 20   
    torch.manual_seed(10)  
    # R = torch.Tensor(N, n).uniform_(0, 10)   
    # W  = torch.Tensor(N, n-1).uniform_(-15, 15) 
    R = torch.Tensor(N, n).uniform_(0, 5)   
    W  = torch.Tensor(N, n-1).uniform_(-10, 10) 
    X = torch.cat([R,W],1) 
    Y = torch.zeros_like(X)       
    L = torch.eye(n)-torch.ones([n,n])/n
    for i in range(N):
        x = X[i,:]
        Y[i,:] = f_value(x,L)
        if i%100:
            print(i)
    torch.save({'X':X,'Y':Y},'./neural_sde/stuart/n_20/20_train_data_small.pt')

# Joint trajcetories on two adjacent time intervals
def cat_data(path0='./neural_sde/stuart/n_20/20_original_data_cat.pt',path1='./neural_sde/stuart/n_20/20_original_data.pt',path2='./neural_sde/stuart/n_20/20_original_data_add.pt'):
    X = torch.load(path1)
    Y = torch.load(path2)
    X = X['X'][0:80001:10]
    Y = Y['X']
    torch.save({'X':torch.cat([X,Y[1:,:]],0)},path0)

# Get the controlled trajectory for \rho,\theta
def diff_to_orig(n,path1='./neural_sde/stuart/n_20/20_original_data.pt',path2='./neural_sde/stuart/n_20/20_test_data.pt'):
    X = torch.load(path1) 
    Y = torch.load(path2)
    orig_data = X['X']
    trans_data = Y['X']
    Wn = orig_data[:,-1:]
    R = trans_data[:,:n]
    dW = trans_data[:,n:]
    R = R+1
    W = torch.cat([dW,Wn],1).T
    T = torch.eye(len(W))
    for i in range(len(T)):
        for k in range(len(T)):
            if k>i:
                T[i,k]=1.0
    orig_W = torch.mm(T,W)
    return torch.cat([R,orig_W.T],1)


if __name__ == '__main__':
    cat_data('./data/stuart/20_original_data_cat.pt','./data/stuart/20_original_data.pt','./data/stuart/20_original_data_add.pt')
    cat_data('./data/stuart/20_test_data_cat.pt','./data/stuart/20_test_data_try.pt','./data/stuart/20_test_data_try_add.pt')
    generate()