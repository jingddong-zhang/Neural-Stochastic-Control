import numpy as np
from cvxopt import solvers,matrix
import matplotlib.pyplot as plt
import torch
import seaborn as sns


class ControlNet(torch.nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(ControlNet,self).__init__()
        torch.manual_seed(2)
        self.layer1=torch.nn.Linear(n_input,n_hidden)
        self.layer2=torch.nn.Linear(n_hidden,n_hidden)
        self.layer3=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        sigmoid=torch.nn.ReLU()
        h_1=sigmoid(self.layer1(x))
        h_2=sigmoid(self.layer2(h_1))
        out=self.layer3(h_2)
        return out

def qp(x1,x2,epi=0.1,p=10.0):
    P = matrix(np.diag([2.0,2.0,2*p]))
    q = matrix([0.0,0.0,0.0])
    G = matrix(np.array([[x1,x2,-1.0]]))
    h = matrix([(-3.0*x1+2.15*x2)**2/2-x2**2-(x1**2+x2**2)/(2*epi)]) # 在Lie算子里加入V/epi项
    # h = matrix([(-3.0*x1+2.15*x2)**2/2-x2**2])
    solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h)  # 调用优化函数solvers.qp求解
    u =np.array(sol['x'])
    return u

def osqp(x1,x2,epi=0.1,p=10.0):
    P = matrix(np.diag([2.0,2.0,2*p]))
    q = matrix([0.0,0.0,0.0])
    G = matrix(np.array([[3*x1+x2,x1+3*x2,-1.0]]))
    h = matrix([x1**2+x1*x2+2*x2**2-(3*x1**2+2*x1*x2+3*x2**2)/(2*epi)-3*(-3.0*x1+2.15*x2)**2/2])
    solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h)  # 调用优化函数solvers.qp求解
    u =np.array(sol['x'])
    return u

model = ControlNet(2,6,2)
model.load_state_dict(torch.load('icnn_net.pkl'))

def harmonic(n,dt,case):
    x0 = np.array([-2.0,2.0])
    X = np.zeros([n,2])
    X[0,:]=x0
    z = np.random.normal(0,1,n)
    for i in range(n-1):
        x1,x2 = X[i,:]
        if case != 3:
            if case == 0:
                u1,u2,d = np.zeros(3)
            if case == 1:
                u1,u2,d = qp(x1,x2)
            if case == 2:
                u1,u2,d=osqp(x1,x2)
            X[i+1,0] = x1 + (x2+u1)*dt
            X[i+1,1] = x2 + (-x1-x2+u2)*dt+(-3*x1+2.15*x2)*np.sqrt(dt)*z[i]
        if case == 3:
            with torch.no_grad():
                u = model(torch.from_numpy(X[i,:]).to(torch.float32))
                u = u.detach().numpy()
                u1,u2 = u[0],u[1]
            X[i+1,0]=x1+(x2)*dt + np.sqrt(dt)*z[i]*u1*x1
            X[i+1,1]=x2+(-x1-x2)*dt+(-3*x1+2.15*x2+u2*x2)*np.sqrt(dt)*z[i]
        if i%3000 == 0:
            print(i,u1,u2)
    return X

n = 4000
dt = 0.00001
font_size=20
X = np.zeros([10,n,2])
# for i in range(10):
#     np.random.seed(20*i)
#     X[i,:] = harmonic(n,dt,3)
# # np.save('qp.npy',X)
# # X = np.load('ES.npy')
# plt.plot(np.arange(n),np.mean(X[:,:,0],axis=0))
# plt.plot(np.arange(n),np.mean(X[:,:,1],axis=0))
def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass

colors = [
    [233/256,	110/256, 236/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    [255/255, 165/255, 0],
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [107/256,	161/256,255/256], # #6ba1ff
    [0.6, 0.4, 0.8], # amethyst
    [0.0, 0.0, 1.0], # ao
    [0.55, 0.71, 0.0], # applegreen
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
]
colors = np.array(colors)

def plot1(alpha=0.1):
    X1 = np.load('ES.npy')
    X1 = X1[:, 0:40000:10, :]
    X2 = np.load('qp.npy')[:, :4000, :]
    X3 = np.load('osqp.npy')[:, :4000, :]
    X4 = np.load('lqr.npy')[:, :4000, :]
    plt.subplot(144)
    plt.fill_between(np.arange(n), np.mean(X1[:, :, 0], 0) - np.std(X1[:, :, 0], 0),
                     np.mean(X1[:, :, 0], 0) + np.std(X1[:, :, 0], 0),
                     color='r', alpha=alpha)
    plt.plot(np.arange(n), np.mean(X1[:, :, 0], axis=0), color='r', label=r'$x_1$')
    plt.fill_between(np.arange(n), np.mean(X1[:, :, 1], 0) - np.std(X1[:, :, 1], 0),
                     np.mean(X1[:, :, 1], 0) + np.std(X1[:, :, 1], 0),
                     color='r', alpha=alpha)
    plt.plot(np.arange(n), np.mean(X1[:, :, 1], axis=0), color='orange', label=r'$x_2$')
    plt.xticks([0, 2000, 4000], [0, 0.2, 0.4])
    plt.xlabel(r'$t$', fontsize=font_size)
    plt.ylabel(r'$x_1$', fontsize=font_size)
    plt.ylim(-4, 4.0)
    plt.legend(loc=4)
    plt.title('ES+ICNN', fontsize=font_size)
    plot_grid()

    plt.subplot(142)
    plt.fill_between(np.arange(n), np.mean(X2[:, :, 0], 0) - np.std(X2[:, :, 0], 0),
                     np.mean(X2[:, :, 0], 0) + np.std(X2[:, :, 0], 0),
                     color='b', alpha=alpha)
    plt.plot(np.arange(n), np.mean(X2[:, :, 0], axis=0), color='r', label=r'$x_1$')
    plt.fill_between(np.arange(n), np.mean(X2[:, :, 1], 0) - np.std(X2[:, :, 1], 0),
                     np.mean(X2[:, :, 1], 0) + np.std(X2[:, :, 1], 0),
                     color='b', alpha=alpha)
    plt.plot(np.arange(n), np.mean(X2[:, :, 1], axis=0), color='orange', label=r'$x_2$')
    plt.xticks([0, 2000, 4000], [0, 0.2, 0.4])
    plt.xlabel(r'$t$', fontsize=font_size)
    plt.ylabel(r'$x_1$', fontsize=font_size)
    plt.ylim(-4, 4.0)
    plt.legend(loc=4)
    plt.title('HDSCLF',fontsize=font_size)
    plot_grid()

    plt.subplot(143)
    plt.fill_between(np.arange(n), np.mean(X3[:, :, 0], 0) - np.std(X3[:, :, 0], 0),
                     np.mean(X3[:, :, 0], 0) + np.std(X3[:, :, 0], 0),
                     color='g', alpha=alpha)
    plt.plot(np.arange(n), np.mean(X3[:, :, 0], axis=0), color='r', label=r'$x_1$')
    plt.fill_between(np.arange(n), np.mean(X3[:, :, 1], 0) - np.std(X3[:, :, 1], 0),
                     np.mean(X3[:, :, 1], 0) + np.std(X3[:, :, 1], 0),
                     color='g', alpha=alpha)
    plt.plot(np.arange(n), np.mean(X3[:, :, 1], axis=0), color='orange', label=r'$x_2$')
    plt.xticks([0, 2000, 4000], [0, 0.2, 0.4])
    plt.xlabel(r'$t$', fontsize=font_size)
    plt.ylabel(r'$x_1$', fontsize=font_size)
    plt.ylim(-4, 4.0)
    plt.legend(loc=4)
    plt.title('BALSA', fontsize=font_size)
    plot_grid()

    plt.subplot(141)
    plt.fill_between(np.arange(n), np.mean(X4[:, :, 0], 0) - np.std(X4[:, :, 0], 0),
                     np.mean(X4[:, :, 0], 0) + np.std(X4[:, :, 0], 0),
                     color='orange', alpha=alpha)
    plt.plot(np.arange(n), np.mean(X4[:, :, 0], axis=0), color='r', label=r'$x_1$')

    plt.fill_between(np.arange(n), np.mean(X4[:, :, 1], 0) - np.std(X4[:, :, 1], 0),
                     np.mean(X4[:, :, 1], 0) + np.std(X4[:, :, 1], 0),
                     color='orange', alpha=alpha)
    plt.plot(np.arange(n), np.mean(X4[:, :, 1], axis=0), color='orange', label=r'$x_2$')
    plt.xticks([0, 2000, 4000], [0, 0.2, 0.4])
    plt.xlabel(r'$t$', fontsize=font_size)
    plt.ylabel(r'$x_1$', fontsize=font_size)
    plt.ylim(-4, 4.0)
    plt.legend(loc=4)
    plt.title('LQR', fontsize=font_size)
    plot_grid()




def plot2(alpha=0.1):
    X1 = np.load('ES.npy')
    X1 = X1[:,0:40000:10,:]
    X2 = np.load('qp.npy')[:,:4000,:]
    X3 = np.load('osqp.npy')[:,:4000,:]
    X4 = np.load('lqr.npy')[:,:4000,:]
    plt.subplot(121)
    plt.fill_between(np.arange(n),np.mean(X1[:,:,0],0)-np.std(X1[:,:,0],0),np.mean(X1[:,:,0],0)+np.std(X1[:,:,0],0),
                     color=colors[0],alpha=alpha)
    plt.plot(np.arange(n),np.mean(X1[:,:,0],axis=0),color=colors[0],label='ES+ICNN')
    plt.fill_between(np.arange(n),np.mean(X2[:,:,0],0)-np.std(X2[:,:,0],0),np.mean(X2[:,:,0],0)+np.std(X2[:,:,0],0),
                     color=colors[1],alpha=alpha)
    plt.plot(np.arange(n),np.mean(X2[:,:,0],axis=0),color=colors[1],label='HDSCLF')
    plt.fill_between(np.arange(n),np.mean(X3[:,:,0],0)-np.std(X3[:,:,0],0),np.mean(X3[:,:,0],0)+np.std(X3[:,:,0],0),
                     color=colors[2],alpha=alpha)
    plt.plot(np.arange(n),np.mean(X3[:,:,0],axis=0),color=colors[2],label='BALSA')
    plt.fill_between(np.arange(n),np.mean(X4[:,:,0],0)-np.std(X4[:,:,0],0),np.mean(X4[:,:,0],0)+np.std(X4[:,:,0],0),
                     color=colors[5],alpha=alpha)
    plt.plot(np.arange(n),np.mean(X4[:,:,0],axis=0),color=colors[5],label='LQR')
    plt.xticks([0,2000,4000],[0,0.2,0.4], fontsize=font_size)
    plt.xlabel('Time',fontsize=font_size)
    plt.ylabel(r'$x_1$',fontsize=font_size)
    plt.yticks([-3,0,3],fontsize=font_size)
    plt.ylim(-3,3.0)
    # plt.legend(loc=4, fontsize=font_size*0.6,)
    # plt.legend(fontsize=font_size * 0.7, ncol=4, bbox_to_anchor=(1.5, 1.1))

    plot_grid()
    plt.subplot(122)
    plt.fill_between(np.arange(n),np.mean(X1[:,:,1],0)-np.std(X1[:,:,1],0),np.mean(X1[:,:,1],0)+np.std(X1[:,:,1],0),
                     color=colors[0],alpha=alpha)
    plt.plot(np.arange(n),np.mean(X1[:,:,1],axis=0),color=colors[0],label='ES+ICNN')
    plt.fill_between(np.arange(n),np.mean(X2[:,:,1],0)-np.std(X2[:,:,1],0),np.mean(X2[:,:,1],0)+np.std(X2[:,:,1],0),
                     color=colors[1],alpha=alpha)
    plt.plot(np.arange(n),np.mean(X2[:,:,1],axis=0),color=colors[1],label='HDSCLF')
    plt.fill_between(np.arange(n),np.mean(X3[:,:,1],0)-np.std(X3[:,:,1],0),np.mean(X3[:,:,1],0)+np.std(X3[:,:,1],0),
                     color=colors[2],alpha=alpha)
    plt.plot(np.arange(n),np.mean(X3[:,:,1],axis=0),color=colors[2],label='BALSA')
    plt.fill_between(np.arange(n),np.mean(X4[:,:,1],0)-np.std(X4[:,:,1],0),np.mean(X4[:,:,1],0)+np.std(X4[:,:,1],0),
                     color=colors[5],alpha=alpha)
    plt.plot(np.arange(n),np.mean(X4[:,:,1],axis=0),color=colors[5],label='LQR')
    plt.xticks([0,2000,4000],[0,0.2,0.4], fontsize=font_size)
    # plt.legend(loc=1, fontsize=font_size*0.6)
    plt.xlabel('Time',fontsize=font_size)
    plt.ylabel(r'$x_2$',fontsize=font_size)
    plt.yticks([ 0, 6], fontsize=font_size)
    plt.ylim(-1,6)
    plot_grid()


# plot1()
plot2()
plt.show()