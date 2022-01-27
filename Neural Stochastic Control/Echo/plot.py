import numpy as np
import matplotlib.pyplot as plt
#Use latex font
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# matplotlib.rcParams['text.usetex'] = True
font_size = 15
def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)

def plot_trajec(L):
    mean_data = np.mean(L,0)
    std_data  = np.std(L,0)
    plt.fill_between(np.arange(len(mean_data)),mean_data-std_data,mean_data+std_data,color='r',alpha=0.2)
    plt.plot(np.arange(len(mean_data)),mean_data,color='r',alpha=0.9)
    plt.yticks([])




plt.subplot(271)
X = np.load('./data/Echo/orig_data.npy')[0:40001:10,:]
for i in range(50):
    plt.plot(np.arange(len(X)),X[:,i])
plt.ylabel('Value',fontsize=font_size)
plt.xlabel('Time',fontsize=font_size)
# plt.text(1,4,r'$\textbf{Tanh}$',rotation=90,fontsize=font_size)
plot_grid()
plt.title('Original',fontsize=font_size)

plt.subplot(272)
X = np.load('./data/Echo/tanh_data.npy')[6,0:50001:10,:]
for i in range(50):
    plt.plot(np.arange(len(X)),X[:,i])
plt.ylim(-2,2)
plt.yticks([-2,-1,0,1,2])
plt.xticks([0,2000,4000],[0,0.02,0.04])
plot_grid()
plt.title('Controlled',fontsize=font_size)


plt.subplot(278)
X = np.load('./data/Echo/relu_orig_data.npy')[0:40001:10,:]
for i in range(50):
    plt.plot(np.arange(len(X)),X[:,i])
plt.ylabel('Value',fontsize=font_size)
plt.xlabel('Time',fontsize=font_size)
plot_grid()

plt.subplot(279)
X = np.load('./data/Echo/relu_data.npy')[6,0:50001:10,:]
for i in range(50):
    plt.plot(np.arange(len(X)),X[:,i])
plt.ylim(-2,2)
plt.yticks([-2,-1,0,1,2])
plt.xticks([0,2000,4000],[0,0.02,0.04])
plot_grid()

for i in range(5):
    plt.subplot(2,7,i+3)
    X = np.load('./data/Echo/tanh_data.npy')[np.delete(np.arange(10),1),0:5001:10,:]
    plot_trajec(X[:,:,i*10+9])
    plt.yticks([-2,-1,0,1,2],[])
    plt.ylim(-2,2)
    plt.xticks([0,200,400],[0,0.002,0.004])
    plot_grid()
    plt.title(r'$x_{10}$',fontsize=font_size)

for i in range(5):
    plt.subplot(2,7,7+i+3)
    X = np.load('./data/Echo/relu_data.npy')[np.delete(np.arange(10),1),0:5001:10,:]
    plot_trajec(X[:,:,i*10+9])
    plt.yticks([-2,-1,0,1,2],[])
    plt.ylim(-2,2)
    plt.xticks([0,200,400],[0,0.002,0.004])
    plot_grid()



plt.show()