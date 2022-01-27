import numpy as np
import torch

data = torch.load('./data/harmonic/data_long.pt')

# Calculate the data in table1
def L2_norm(st,a):
    Y = data[st][torch.tensor([0,1,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19]),:,:]
    Y = Y.detach().numpy()
    X = np.linalg.norm(Y,axis=2)
    Z = np.mean(X,0)
    index = np.where(Z<0.05)
    print('{} min :'.format(a),np.min(Z))
    print('{} convergence time of 0.05:'.format(a), format(index[0][0]*1e-5,'.3f'))

L2_norm('Y','ICNN')
L2_norm('Z','Quad')
L2_norm('W','AS')

