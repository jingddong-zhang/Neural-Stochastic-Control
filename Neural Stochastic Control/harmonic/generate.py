import numpy as np
import math
import torch
import numpy as np
import timeit 
from AS import *
from Control_Nonlinear_Icnn import *

start = timeit.default_timer()
# Harmonic linear oscillator
model = Net(D_in,H1,D_out)


# Generate trajectory with nonlinaer AS control
def algo2(z,X,N,dt):
    model = Net(D_in,H1,D_out)
    model.load_state_dict(torch.load('./data/harmonic/algo2_net.pkl'))
    beta = 0.5
    for i in range(N):
        x = X[i]
        with torch.no_grad():
            u = model(torch.tensor(x))
            # -model((torch.tensor([0.0,0.0])))
        x1,x2 = x[0],x[1]
        new_x1 = x1 + x2*dt + math.sqrt(dt)*z[i]*u[0]*x1
        new_x2 = x2 + (-x1-2*beta*x2)*dt + z[i]*(-3*x1+2.15*x2+u[1]*x2)*math.sqrt(dt)
        # new_x1 = x1 + x2*dt + math.sqrt(dt)*z[i]*u[0]
        # new_x2 = x2 + (-x1-2*beta*x2)*dt + z[i]*(-3*x1+2.15*x2+u[1])*math.sqrt(dt)
        X.append([new_x1,new_x2])
    X = torch.tensor(X)
    return X

# Generate trajectory with  linear ES(+Quadratic) control
def algo1(z,X,N,dt,a,b,c,d):
    beta = 0.5
    for i in range(N):
        x = X[i]
        x1,x2 = x[0],x[1]
        new_x1 = x1 + x2*dt + math.sqrt(dt)*z[i]*(a*x1+b*x2)
        new_x2 = x2 + (-x1-2*beta*x2)*dt + z[i]*(-3*x1+2.15*x2+c*x1+d*x2)*math.sqrt(dt)
        X.append([new_x1,new_x2])
    X = torch.tensor(X)
    return X
# Generate trajectory with  nonlinear ES(+ICNN) control
def algo_icnn(z,X,N,dt):
    model2 = ControlNet(D_in,H1,D_out)
    model2.load_state_dict(torch.load('./data/harmonic/icnn_net.pkl'))
    beta = 0.5
    for i in range(N):
        x = X[i]
        with torch.no_grad():
            u = model2(torch.tensor(x))
        x1,x2 = x[0],x[1]
        new_x1 = x1 + x2*dt + math.sqrt(dt)*z[i]*u[0]*x1
        new_x2 = x2 + (-x1-2*beta*x2)*dt + z[i]*(-3*x1+2.15*x2+u[1]*x2)*math.sqrt(dt)
        X.append([new_x1,new_x2])
    X = torch.tensor(X)
    return X 


def generate(m,N,dt):
    X,Y,Z,W = torch.zeros(m,N+1,2),torch.zeros(m,N+1,2),torch.zeros(m,N+1,2),torch.zeros(m,N+1,2)
    for r in range(m):
        # x0 = [0.3,0.5] #Fixed initial
        x0 = [np.random.uniform(-2,2),np.random.uniform(-2,2)] #random initial
        np.random.seed(12*r)
        z = np.random.normal(0,1,N)
        X[r,:] = algo1(z,[x0],N,dt,0,0,0,0) # Without control
        Y[r,:] = algo_icnn(z,[x0],N,dt)
        Z[r,:] = algo1(z,[x0],N,dt,1.726,-0.4946,2.0548,0.3159) #Quadratic 2.2867,0.3492,1.593,-0.4191 61.6973088
        W[r,:] = algo2(z,[x0],N,dt)
        print(r)
    return {'X':X,'Y':Y,'Z':Z,'W':W}



# Sample numbers, Iterations in per trajectory, and sample time interval : 20,400000,0.00001
torch.save(generate(20,400000,0.00001),'./data/harmonic/data_long.pt')
torch.save(generate(20,400000,0.00001),'./data/harmonic/data_long_random.pt')
stop = timeit.default_timer() 
print('total time:',stop-start)
