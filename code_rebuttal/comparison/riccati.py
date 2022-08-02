import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#由题目定义矩阵
A = np.matrix([[0, 1.0], [-1.0, -1.0]])
AT = np.matrix([[0,-1.0], [1, -1.0]])
B = np.matrix([[1.0,0.0], [0.0,1.0]])
BT = np.matrix([[1.0,0.0], [0.0,1.0]])
F = np.matrix([[1, 0], [0, 2]])
Q = np.matrix([[20.0, 0.0], [0.0, 20.0]])
R = np.matrix([[1.0, 0.0], [0.0, 1.0]])
RN = 2

#根据边界条件给定的值
step_num = 200
t = 3
step = -t / step_num
P = F

# 定义黎卡提方程
def Ricatti_P(t, P):
    f = -(P * A + A.T * P - P * B * R.I * B.T * P + Q)
    return f

ys_0, ys_1, ys_2, ys_3 = [], [], [], []
ts = []
while t > 0:
    t += step
    k1 = step * Ricatti_P(t, P)
    k2 = step * Ricatti_P(t + step * 0.5, P + k1 * step * 0.5)
    k3 = step * Ricatti_P(t + step * 0.5, P + k2 * step * 0.5)
    k4 = step * Ricatti_P(t + step, P + k3 * step)
    P = P + (k1 + k2 * 2 + k3 * 2 + k4)/6
    P = np.array(P)
    ts.append(t)
    ys_0.append(P[0][0])
    ys_1.append(P[0][1])
    ys_2.append(P[1][0])
    ys_3.append(P[1][1])
    # print(ys_0)
print(P)
# P = np.matrix([[3, 1], [1, 3]])
# print(P * A + A.T * P)
