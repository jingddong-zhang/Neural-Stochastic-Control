from os import stat
import numpy as np
import math
import torch
import timeit 
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import odeint
import numpy as np

np.random.seed(10)

class ControlNet(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(ControlNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out


D_in = 2            # input dimension
H1 = 6             # hidden dimension
D_out = 2  
inverted_model = ControlNet(D_in,H1,D_out)
inverted_model.load_state_dict(torch.load('./neural_sde/hyper_b/b_2.2.pkl'))


# ang = torch.zeros([5,1]) #initial angle
# vel = torch.zeros([5,1]) #initial velocity
# for i in range(5):
#     x0 = np.random.uniform(-6,6,2)
#     ang[i,0] = x0[0]
#     vel[i,0] = x0[1]

def invert_pendulum(state0, t):
    state0 = state0.flatten()
    G = 9.81  # gravity
    L = 0.5   # length of the pole 
    m = 0.15  # ball mass
    b = 0.1   # friction
    def f(state,t):
        x, y = state  # unpack the state vector
        return y, G*np.sin(x)/L +(-b*y)/(m*L**2) # derivatives
    states = odeint(f, state0, t)
    return states.transpose()

#生成控制轨道数据
set_state0 = torch.tensor([[-5.0,5.0],[-3.0,4.0],[-1.0,3.0],[1.0,-3.0],[3.0,-4.0],[5.0,-5.0]])
def control_data(set_state0,M=6,N=20000,dt=0.00001):
    start = timeit.default_timer()
    torch.manual_seed(6)  
    X1,X2 = torch.zeros([M,N]),torch.zeros([M,N])
    for r in range(M):
        G = 9.81  # gravity
        L = 0.5   # length of the pole 
        m = 0.15  # ball mass
        b = 0.1
        z1 = torch.randn(N)
        z2 = torch.randn(N)
        # X1[r,0] = ang[r,0]
        # X2[r,0] = vel[r,0]
        X1[r,0] = set_state0[r,0]
        X2[r,0] = set_state0[r,1]
        for i in range(N-1):
            x1 = X1[r,i]
            x2 = X2[r,i]
            u = inverted_model(torch.tensor([x1,x2]))
            new_x1 = x1 + x2*dt + x1*u[0]*z1[i]*math.sqrt(dt)
            new_x2 = x2 + (G*math.sin(x1)/L - b*x2/(m*L**2))*dt + x2*u[1]*z2[i]*math.sqrt(dt)
            X1[r,i+1] = new_x1
            X2[r,i+1] = new_x2
       
        print('{} done'.format(r))
    orig_data = {'X1':X1,'X2':X2}
    torch.save(orig_data,'./neural_sde/inverted_ROA/control_data.pt')
    stop = timeit.default_timer()
    print(stop-start)

def control_trajectory(ax,title,path='./neural_sde/inverted_ROA/control_data.pt'):
    data = torch.load(path)
    # X = data['X'].clone().detach()
    X1 = data['X1'].clone().detach()
    # X2 = data['X2']
    for i in range(len(X1)):
        # x = X[i,:].numpy()
        # m = np.max(x)
        # index = np.argwhere(x == m )
        # sample_length = int(index[0])
        L = np.arange(len(X1[0,:3000]))
        plt.plot(L[0],X1[i,0],marker='*',markersize=8,color=cm.Accent(i*2))
        plt.plot(L,X1[i,:3000],linestyle='--',color=cm.Accent(i*2),alpha=0.45)

    L1 = plt.axhline(y=0.0,ls="--",linewidth=1.5,color="green")#添加水平直线
    L2 = plt.axhline(y=math.pi,ls="--",linewidth=1.5,color="r")
    L3 = plt.axhline(y=-math.pi,ls="--",linewidth=1.5,color="b")
    ax.add_artist(L1)
    ax.add_artist(L2)
    ax.add_artist(L3)
    plt.legend([L1,L2,L3],[r'$\theta=0$',r'$\theta=\pi$',r'$\theta=-\pi$'],loc='upper right',borderpad=0.05, labelspacing=0.05)
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel(r'$\theta$')


def f(y) :
    #parameters
    G = 9.81 
    L = 0.5  
    m = 0.15  
    b = 0.1   
    x1,x2 = y    
    dydt =[x2,  (m*G*L*np.sin(x1) - b*x2) / (m*L**2)]
    return dydt

#绘制向量场
def Plotflow(Xd, Yd):
    # Plot phase plane 
    DX, DY = f([Xd, Yd])
    DX=DX/np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY=DY/np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
    plt.streamplot(Xd,Yd,DX,DY, color=('gray'), linewidth=0.5,
                  density=0.6, arrowstyle='-|>', arrowsize=1.5)
    
if __name__ == '__main__':
    control_data(set_state0,6,20000,0.0001)