import numpy as np
import sys
sys.path.append('./data')


# generate the data of table2
def L2_norm(a,case):
    # the case for P_1 
    if case == 0:
        Y = np.load('./data/{}_data_P1_Q2_20.npy'.format(a))
        ind = np.delete(np.arange(20),np.array([1,3,11,15]))
        Y = Y[ind,:].astype('float64')
        Y = Y.astype('float64')
        X = np.linalg.norm(Y,axis=1)
        Z = np.mean(X,0)
        index = np.where(Z<1e-10)
        print('{} convergence time of 1e-10:'.format(a), format(index[0][0]*5e-5,'.3f'))
        print('{} min :'.format(a),np.min(Z))
    # the case for P_2 
    else:
        e = np.array([[6*np.sqrt(2)],[6*np.sqrt(2)],[27]])
        Y = np.load('./neural_sde/calculate/{}_data_P2_Q2_20.npy'.format(a))
        ind = np.delete(np.arange(20),np.array([10,12]))
        Y = Y[ind,:].astype('float64')
        Y = Y.astype('float64')
        for i in range(len(Y)):
            Y[i,:] = Y[i,:]-e
        X = np.linalg.norm(Y,axis=1)
        Z = np.mean(X,0)
        index = np.where(Z<0.02)
        print('{} convergence time of 0.02:'.format(a), format(index[0][0]*1e-4,'.3f'))
        print('{} min :'.format(a),np.min(Z))
L2_norm('icnn',0)
L2_norm('quad',0)
L2_norm('icnn',1)
L2_norm('quad',1)

