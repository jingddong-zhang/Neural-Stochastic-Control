import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from networkx.generators.random_graphs import barabasi_albert_graph, erdos_renyi_graph



def initial_W(shape, low_bound, up_bound):
    return np.random.uniform(low_bound, up_bound, size=shape)


def generate_A(shape, rho, D_r):
    '''
    :param shape: Shape of matrix A （D_r, D_r）
    :param rho:   Spectrum radius of matrix A
    :param D_r:  Dimension of matirx A
    :return:  Generated matrix A
    '''
    G = erdos_renyi_graph(D_r, 6 / D_r, seed=2)   # Generate ER graph with D_r nodes, the connection probability is p = 3 /D_r
    degree = [val for (node, val) in G.degree()]
    print('average degree:', sum(degree) / len(degree))
    G_A = nx.to_numpy_matrix(G)  # Transform the graph to the connection matrix A
    index = np.where(G_A > 0)  # Find the position where has an edge
    
    res_A = np.zeros(shape)

    a = 0.3
    res_A[index] = initial_W([len(index[0]), ], 0, a)  # Sample value for edge from Uniform[0,a]
    max_eigvalue = np.real(np.max(LA.eigvals(res_A)))  # Calculate the largest eigenvalue of A
    print('before max_eigvalue:{}'.format(max_eigvalue))
    res_A = res_A / abs(max_eigvalue) * rho  # Adjust spectrum radius of A to rho
    max_eigvalue = np.real(np.max(LA.eigvals(res_A)))
    print('after max_eigvalue:{}'.format(max_eigvalue))

    return res_A, max_eigvalue


rho = 2.0
D_r = 50
res_A, max_eigval = generate_A(shape=(D_r, D_r), rho=rho, D_r=D_r)
np.save('./data/Echo/A_50.npy', res_A)