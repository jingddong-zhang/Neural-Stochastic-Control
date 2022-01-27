from turtle import color
import numpy as np
import math
import torch
import timeit 
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# matplotlib.rcParams['text.usetex'] = True

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
np.random.seed(10)

class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
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

D_in = 3          
H1 = 10             
D_out = 3  

model = Net(D_in,H1,D_out)
# set_state0 = torch.tensor([[3.0,5.0,6.0]])
def control_data(model,random_seed,set_state0,N=20000,dt=0.00001):
    start = timeit.default_timer()
    torch.manual_seed(random_seed)  
    X = torch.zeros([3,N])
    z = torch.randn(N)
    X[0,0] = set_state0[0,0]
    X[1,0] = set_state0[0,1]
    X[2,0] = set_state0[0,2]
    for i in range(N-1):
        x1 = X[0,i]
        x2 = X[1,i]
        x3 = X[2,i]
        with torch.no_grad():
            u = model(torch.tensor([x1,x2,x3]))
        new_x1 = x1 + 10*(x2-x1)*dt +      x1*u[0]*z[i]*math.sqrt(dt)
        new_x2 = x2 + (x1*(28-x3)-x2)*dt + x2*u[1]*z[i]*math.sqrt(dt)
        new_x3 = x3 + (x1*x2-8/3*x3)*dt +  x3*u[2]*z[i]*math.sqrt(dt)
        X[0,i+1] = new_x1
        X[1,i+1] = new_x2
        X[2,i+1] = new_x3
    stop = timeit.default_timer()
    print(stop-start)
    return X

def modify_control_data(model,random_seed,set_state0,N=20000,dt=0.00001):
    start = timeit.default_timer()
    torch.manual_seed(random_seed)  
    X = torch.zeros([3,N])
    z = torch.randn(N)
    e = torch.tensor([6.0*math.sqrt(2), 6.0*math.sqrt(2) , 27.0])
    e1,e2,e3=e
    X[0,0] = set_state0[0,0]
    X[1,0] = set_state0[0,1]
    X[2,0] = set_state0[0,1]
    for i in range(N-1):
        x1 = X[0,i]
        x2 = X[1,i]
        x3 = X[2,i]
        with torch.no_grad():
            u = model(torch.tensor([x1-e1,x2-e2,x3-e3]))
        new_x1 = x1 + 10*(x2-x1)*dt +      (x1-e1)*u[0]*z[i]*math.sqrt(dt)
        new_x2 = x2 + (x1*(28-x3)-x2)*dt + (x2-e2)*u[1]*z[i]*math.sqrt(dt)
        new_x3 = x3 + (x1*x2-8/3*x3)*dt +  (x3-e3)*u[2]*z[i]*math.sqrt(dt)
        X[0,i+1] = new_x1
        X[1,i+1] = new_x2
        X[2,i+1] = new_x3
    stop = timeit.default_timer()
    print(stop-start)
    return X

def original_data(set_state0,N=50000,dt=0.001):
    start = timeit.default_timer()
    X = torch.zeros([3,N])
    X[0,0] = set_state0[0,0]
    X[1,0] = set_state0[0,1]
    X[2,0] = set_state0[0,1]
    for i in range(N-1):
        x1 = X[0,i]
        x2 = X[1,i]
        x3 = X[2,i]
        new_x1 = x1 + 10*(x2-x1)*dt 
        new_x2 = x2 + (x1*(28-x3)-x2)*dt
        new_x3 = x3 + (x1*x2-8/3*x3)*dt 
        X[0,i+1] = new_x1
        X[1,i+1] = new_x2
        X[2,i+1] = new_x3
    stop = timeit.default_timer()
    print(stop-start)
    torch.save(X,'./data/Lorenz/original_data.pt')
    return X


def plot_original_orbit():
    fig = plt.figure()
    X = torch.load('./data/Lorenz/original_data.pt')[:,0:50000:10]
    x1,x2,x3=X[0,:],X[1,:],X[2,:]
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax = fig.add_subplot(111,projection = '3d')
    ax.plot3D(x1,x2,x3,color=[1.0, 0.8, 0.6])
    ax.plot3D(0,0,0,marker='*',label=r'$P_1$',color=colors[0])
    ax.plot3D(6*math.sqrt(2),6*math.sqrt(2),27,marker='*',label=r'$P_2$',color=colors[3])
    ax.plot3D(-6*math.sqrt(2),-6*math.sqrt(2),27,marker='*',label=r'$P_3$',color=colors[2])
    plt.legend()

def orbit1(ax,path1,P1):
    # fig = plt.figure()
    Q1 =np.load('./data/Lorenz/{}_data_{}_Q1.npy'.format(path1,P1))[0,:,0:100000:10]
    Q2 =np.load('./data/Lorenz/{}_data_{}_Q2.npy'.format(path1,P1))[0,:,0:100000:10]
    Q3 =np.load('./data/Lorenz/{}_data_{}_Q3.npy'.format(path1,P1))[0,:,0:100000:10]
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    # ax = fig.add_subplot(111,projection = '3d')
    ax.plot3D(Q1[0,:],Q1[1,:],Q1[2,:],color=colors[4],alpha=0.5)
    ax.plot3D(Q2[0,:],Q2[1,:],Q2[2,:],color=colors[5],alpha=0.5)
    ax.plot3D(Q3[0,:],Q3[1,:],Q3[2,:],color=colors[7],alpha=0.5)
    ax.plot3D(0,0,0,marker='*',label=r'$P_1$',markersize=10,color=colors[0])
    # ax.plot3D(6*math.sqrt(2),6*math.sqrt(2),27,marker='*',label=r'$P_2$')
    # ax.plot3D(-6*math.sqrt(2),-6*math.sqrt(2),27,marker='*',label=r'$P_3$')
    ax.plot3D(9,6,8,marker='*',label=r'$Q_1$',markersize=10,color=colors[4])
    ax.plot3D(3,5,6,marker='*',label=r'$Q_2$',markersize=10,color=colors[5])
    ax.plot3D(1,9,2,marker='*',label=r'$Q_3$',markersize=10,color=colors[7])
    # ax.plot3D(8,2,1,marker='^',label=r'$Q_4$')
    ax.set_xlabel(r'$X$')
    # ax.set_xlim(0, 10)  
    ax.set_ylabel(r'$Y$')
    # ax.set_ylim(0, 10)
    ax.set_zlabel(r'$Z$')
    # ax.set_zlim(0, 10)
    plt.legend(fontsize=8,markerscale=0.5,labelspacing=0.05,borderpad=0.1,handlelength=1.0)    

def orbit2(ax,path1,P1):
    # fig = plt.figure()
    Q1 =np.load('./data/Lorenz/{}_data_{}_Q1.npy'.format(path1,P1))[0,:,0:200000:10]
    Q2 =np.load('./data/Lorenz/{}_data_{}_Q2.npy'.format(path1,P1))[0,:,0:200000:10]
    Q3 =np.load('./data/Lorenz/{}_data_{}_Q3.npy'.format(path1,P1))[0,:,0:200000:10]
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    # ax = fig.add_subplot(111,projection = '3d')
    ax.plot3D(Q1[0,:],Q1[1,:],Q1[2,:],color=colors[4],alpha=0.5)
    ax.plot3D(Q2[0,:],Q2[1,:],Q2[2,:],color=colors[5],alpha=0.5)
    ax.plot3D(Q3[0,:],Q3[1,:],Q3[2,:],color=colors[7],alpha=0.5)
    # ax.plot3D(0,0,0,marker='*',label=r'$P_1$',markersize=10)
    ax.plot3D(6*math.sqrt(2),6*math.sqrt(2),27,marker='*',label=r'$P_2$',markersize=10,color=colors[3])
    # ax.plot3D(-6*math.sqrt(2),-6*math.sqrt(2),27,marker='*',label=r'$P_3$')
    ax.plot3D(9,6,8,marker='*',label=r'$Q_1$',markersize=10,color=colors[4])
    ax.plot3D(3,5,6,marker='*',label=r'$Q_2$',markersize=10,color=colors[5])
    ax.plot3D(1,9,2,marker='*',label=r'$Q_3$',markersize=10,color=colors[7])
    ax.set_xlabel(r'$X$')
    # ax.set_xlim(0, 10)  
    ax.set_ylabel(r'$Y$')
    # ax.set_ylim(0, 10)
    ax.set_zlabel(r'$Z$')
    # ax.set_zlim(0, 10)
    plt.legend(fontsize=8,markerscale=0.5,labelspacing=0.05,borderpad=0.1,handlelength=1.0)  
    # plt.legend(loc='upper right',labelspacing=0.1,borderpad=0.2,handlelength=1.2)



def plot_original_tra():
    X = torch.load('./data/Lorenz/original_data.pt')[:,0:40000:10]
    x1,x2,x3=X[0,:],X[1,:],X[2,:]
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.subplot(131)
    plt.xticks([])
    plt.plot(np.arange(len(x1)),x1,label='x',color='r')
    plt.ylabel(r'$x$')
    plt.subplot(132)
    plt.xticks([])
    plt.plot(np.arange(len(x1)),x2,label='y',color='g')
    plt.ylabel(r'$y$')
    plt.subplot(133)
    plt.xticks([0,1000,2000,3000,4000],[0,10,20,30,40])
    plt.plot(np.arange(len(x1)),x3,label='z',color='b')
    plt.ylabel(r'$z$')
    plt.xlabel('Time')

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)

def plot_tra(path1,P1,Q1,length=200000):
    X = np.load('./data/Lorenz/{}_data_{}_{}.npy'.format(path1,P1,Q1))[0,:,0:length:10]
    x1,x2,x3=X[0,:],X[1,:],X[2,:]
    plt.plot(np.arange(len(x1)),x1,label='x',color='r')
    plt.plot(np.arange(len(x1)),x2,label='y',color='g')
    plt.plot(np.arange(len(x1)),x3,label='z',color='b')
    plot_grid()
    plt.legend(loc='upper right',labelspacing=0.1,borderpad=0.2,handlelength=1.2)



def quad_generate(set_state0,m,N,dt,case):
    X = torch.zeros(m,3,N)
    # model.load_state_dict(torch.load('./neural_sde/Lorenz/ES_quad_net_modify_0.pkl'))
    # model.load_state_dict(torch.load('./neural_sde/Lorenz/ES_quad_net_modify_1.pkl'))
    if case == 0:
        model.load_state_dict(torch.load('./data/Lorenz/ES_quad_net_modify_0.pkl'))
        for i in range(m):
            X[i,:] = control_data(model,i*6+2,set_state0,N,dt)
            print(case,i)
        X = X.detach().numpy()
        np.save('./data/Lorenz/quad_data_P1_Q2_20',X)
    else:
        model.load_state_dict(torch.load('./data/Lorenz/ES_quad_net_modify_1.pkl'))
        for i in range(m):
            X[i,:] = modify_control_data(model,i*6+2,set_state0,N,dt)
            print(case,i)
        X = X.detach().numpy()
        np.save('./data/Lorenz/quad_data_P2_Q2_20',X)
    # return X 

def icnn_generate(set_state0,m,N,dt,case):
    X = torch.zeros(m,3,N)
    # model.load_state_dict(torch.load('./neural_sde/Lorenz/ES_icnn_net_100.pkl'))
    # model.load_state_dict(torch.load('./neural_sde/Lorenz/ES_icnn_net_modify_1.pkl'))
    if case == 0:
        model.load_state_dict(torch.load('./data/Lorenz/ES_icnn_net_100.pkl'))
        for i in range(m):
            X[i,:] = control_data(model,i*6+6,set_state0,N,dt)
            print(case,i)
        X = X.detach().numpy()
        np.save('./data/Lorenz/icnn_data_P1_Q2_20',X)
    else:
        model.load_state_dict(torch.load('./data/Lorenz/ES_icnn_net_modify_1.pkl'))
        for i in range(m):
            X[i,:] = modify_control_data(model,i*6+6,set_state0,N,dt)
            print(case,i)
        X = X.detach().numpy()
        np.save('./data/Lorenz/icnn_data_P2_Q2_20',X)
    # return X 





font_size = 15

def plot1():
    fig = plt.figure()
    ax1 = fig.add_subplot(4,4,4,projection = '3d')
    orbit1(ax1,'icnn','P1')
    plt.title('Orbit')

    ax2 = fig.add_subplot(4,4,8,projection = '3d')
    orbit1(ax2,'quad','P1')
    

    ax3 = fig.add_subplot(4,4,12,projection = '3d')
    orbit2(ax3,'icnn','P2')

    ax4 = fig.add_subplot(4,4,16,projection = '3d')
    orbit2(ax4,'quad','P2')
    


def plot2():
    for i in range(3):
        plt.subplot(4,3,i+1)
        plot_tra('icnn','P1','Q{}'.format(i+1),5000)
        plt.xticks([0,200,400],['0','0.02','0.04'])
        plt.title(r'$Q_{}$'.format(i+1),fontsize=font_size)
        if i ==0:
            plt.ylabel(r'$Value$',fontsize=font_size)
            plt.text(0.1,4,r'$ICNN : P_1$',rotation=90,fontsize=font_size)
        if i==1:
            plt.xlabel('Time',fontsize=font_size)
    
    for i in range(3):
        plt.subplot(4,3,3+i+1)
        plot_tra('quad','P1','Q{}'.format(i+1),5000)
        plt.xticks([0,200,400],['0','0.02','0.04'])
        if i==1:
            plt.xlabel('Time',fontsize=font_size)
        if i ==0:
            plt.ylabel(r'$Value$',fontsize=font_size)
            plt.text(0.1,3,r'$Quad : P_1$',rotation=90,fontsize=font_size)
    
    for i in range(3):
        plt.subplot(4,3,6+i+1)
        plot_tra('icnn','P2','Q{}'.format(i+1),200000)
        plt.xticks([0,10000,20000],['0','1.0','2.0'])
        plt.ylim(-10,35)
        if i==1:
            plt.xlabel('Time',fontsize=font_size)
        if i ==0:
            plt.ylabel(r'$Value$',fontsize=font_size)
            plt.text(-0.5,2,r'$ICNN : P_2$',rotation=90,fontsize=font_size)
    
    for i in range(3):
        plt.subplot(4,3,9+i+1)
        plot_tra('quad','P2','Q{}'.format(i+1),200000)
        plt.xticks([0,10000,20000],['0','1.0','2.0'])
        plt.ylim(-10,35)
        if i==1:
            plt.xlabel('Time',fontsize=font_size)
        if i ==0:
            plt.ylabel(r'$Value$',fontsize=font_size)
            plt.text(-0.5,1,r'$Quad : P_2$',rotation=90,fontsize=font_size)





if __name__ == '__main__':
    Q1 = torch.tensor([[9.0,6.0,8.0]]) 
    Q2 = torch.tensor([[3.0,5.0,6.0]])
    Q3 = torch.tensor([[1.0,9.0,2.0]])
    '''
    generate control data
    '''
    icnn_generate(Q2,20,200000,0.00001,0)
    quad_generate(Q2,20,200000,0.00001,0)
    icnn_generate(Q2,20,200000,0.00001,1)
    quad_generate(Q2,20,200000,0.0001,1)
    '''
    Plot figure in Lorenz Experiment
    '''
    # plot1()
    # plot2()
    # original_data(set_state0)
    # plot_original_orbit()
    # plot_original_tra()
    # plt.show()