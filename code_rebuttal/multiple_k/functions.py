import numpy as np
import math
import torch
import timeit 
from scipy import integrate
import matplotlib.pyplot as plt



start = timeit.default_timer()
np.random.seed(1)


class Net(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        return out
    
log_model = Net(1,6,1)
log_model.load_state_dict(torch.load('./data/theta0.9_1d_log_net.pkl'))


N = 100000
dt = 0.00001
m = 20
T = 50
def k_list(N,dt,k,m):
    x0 = [20.0]
    data = torch.zeros([N+1,m])
    for r in range(m):
        np.random.seed(r * 4 + 1)
        X = []
        X.append(x0)
        z = np.random.normal(0,1,N)
        for i in range(N):
            x = X[i][0]
            new_x = x + x*math.log(1+abs(x))*dt + k*x*math.sqrt(dt)*z[i]
            X.append([new_x])
        X = torch.tensor(X)
        data[:,r] = X[:,0]
    return data

def learning_control(N,dt,m):
    x0 = [20.0]
    data = torch.zeros([N+1,m])
    for r in range(m):
        X = []
        X.append(x0)
        np.random.seed(r*4+1)
        z = np.random.normal(0,1,N)
        for i in range(N):
            x = X[i][0]
            k = log_model(torch.tensor([X[i]]))
            new_x = x + x*math.log(1+abs(x))*dt + k[0]*x*math.sqrt(dt)*z[i]
            X.append([new_x])
        X = torch.tensor(X)
        data[:,r] = X[:,0]
        print(r)
    return data

def multiple_k(T,N,dt,m):
    x0 = [50.0]
    data = torch.zeros([T, N + 1, m])
    def generate(k):
        data = torch.zeros([N+1,m])
        for r in range(m):
            X = []
            X.append(x0)
            np.random.seed(r * 4 + 1)
            z = np.random.normal(0, 1, N)
            for i in range(N):
                x = X[i][0]
                new_x = x + x * math.log(1 + abs(x)) * dt + k * x * math.sqrt(dt) * z[i]
                X.append([new_x])
            X = torch.tensor(X)
            data[:, r] = X[:, 0]
            print(r)
        return data
    for j in range(T):
        k = 0.2*(j+1)
        data[j,:]=generate(k)
    torch.save(data,'./data/k_table_x0_20.pt')
    return data




def stopping_time(j):
    data = torch.load('./data/k_table_x0_20_100.pt').numpy()
    X = data[j,:]
    t_x = 0.0
    dt = 0.0001
    for i in range(20):
        norm_x = np.abs(X[:, i])
        ind = np.where(norm_x < 0.1)[0][0] if np.min(norm_x) < 0.1 else int(len(X)) - 1
        t_x += ind*dt
    print(t_x/20)
    return t_x/20

def single_k_energy(j):
    data = torch.load('./data/k_table_x0_20_100.pt').numpy()
    # data = Data['data']
    # X = data[i,:75001,:]
    # N = int(len(X))-1
    X = data[j,:]
    # dt = 0.00001
    dt = 0.00001
    k = ((j+1)*0.2)**2
    gx = k*X**2
    # a = np.linspace(0, dt*N, N+1)
    v_x = 0
    max_norm = 0.0
    for i in range(20):
        norm_x = np.abs(X[:, i])
        ind = np.where(norm_x < 0.1)[0][0] if np.min(norm_x) < 0.1 else int(len(X))-1
        a = np.linspace(0, dt * ind, ind + 1)
        g_x = gx[:,i]
        v_x += integrate.trapz(g_x[0:ind + 1], a)
        max_norm += np.sqrt(np.max(gx))
        # v_x += integrate.trapz(np.array(g_x), a)
        # print(i)
    print(v_x/20,max_norm/20)
    return v_x/20

'''
generate energy_list for different k
'''
T = 50
energy_list = np.zeros(T)
# time_list = np.zeros(T)
for i in range(T):
    energy_list[i] = single_k_energy(i)
    # time_list[i] = stopping_time(i)
# np.save('./data/energy_list',energy_list)
# np.save('./data/time_list',time_list)
# energy_list = np.load('./data/energy_list.npy')
plt.plot(np.arange(T),np.log(energy_list))
# plt.axhline(np.log(1438))
# plt.axhline(0.38)
# plt.show()

# Data = torch.load('./data/20seed_learning_control.pt')
# data = Data['data'].detach().numpy()
# Y = data[0,:][:,np.delete(np.arange(20),15)]# Delete the diverge trajectory due to the dt is not small enough in Euler method
# max_norm = 0.0

# for i in range(19):
#     g_y = (log_model(torch.tensor(Y[:, i]).unsqueeze(1))[:, 0].detach().numpy() * Y[:, i])**2
#     max_norm+=np.sqrt(np.max(g_y))
# print(max_norm)

def k_data():
    endpoint = torch.zeros(T)
    Data = torch.zeros(T,N+1,m)
    for i in range(T):
        k = i*0.2+0.2
        data = k_list(N,dt,k,m)
        endpoint[i] = data[-1].mean()
        Data[i,:] = data
        print(i)
    torch.save({'data':Data,'end':endpoint},'./data/k_table_x0_20.pt')

def learning_data():
    # data = learning_control(200000,dt,10)
    data = learning_control(100000,dt,20)
    # torch.save({'data':data},'./neural_sde/Energy/20_learning_control.pt')
    torch.save({'data':data},'./data/20seed_learning_control.pt')


def k_energy_cost():
    Data = torch.load('k_table.pt')
    data = Data['data'] 
    X = data[29,:75001,:]
    N = 75000
    dt = 0.00001
    gx = 6*2*X**2
    a = np.linspace(0, dt*N, N+1)
    print(a.shape)
    v_x = 0
    for i in range(20):
        g_x = gx[:,i]
        v_x += integrate.trapz(np.array(g_x), a)
        print(i)
    print(v_x/20)

def energy_cost():
    Data = torch.load('./data/20seed_learning_control.pt')
    data = Data['data'].detach().numpy()
    X = data[1,:]
    Y = data[0,:][:,np.delete(np.arange(20),15)]# Delete the diverge trajectory due to the dt is not small enough in Euler method
    N = 100000
    dt = 0.00001
    v_x = 0
    v_y = 0
    # a = np.linspace(0, dt*N, N+1)
    for i in range(Y.shape[1]):
        g_x = 36*X[:,i]**2
        g_y = (log_model(torch.tensor(Y[:,i]).unsqueeze(1))[:,0].detach().numpy()*Y[:,i])**2
        norm_x = np.abs(X[:,i])
        norm_y = np.abs(Y[:,i])
        ind1 = np.where(norm_x<0.1)[0][0]
        ind2 = np.where(norm_y<0.1)[0][0]
        a1 = np.linspace(0, dt*ind1, ind1+1)
        a2 = np.linspace(0, dt*ind2, ind2+1)
        v_x += integrate.trapz(g_x[0:ind1+1], a1)
        v_y += integrate.trapz(g_y[0:ind2+1], a2) 
        print(i)
    print(v_x/20,v_y/19)


# X = multiple_k(T,n,dt,m) # generate data
# k_energy_cost()
# energy_cost()
# learning_data()
# k_data()
# learning_data()
stop= timeit.default_timer()
print('time:',stop-start)

