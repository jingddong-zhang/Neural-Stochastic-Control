import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# matplotlib.rcParams['text.usetex'] = True

font_size=35

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.3, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.3, ls='-', lw=1)


plt.subplot(221)
data = np.load('./neural_sde/Echo/50/k_list.npy')
# norm = np.linalg.norm(data[30,:],axis=2)
# ind = np.where(norm[3,:]<0.1)[0][0]
# print(ind)
end = np.mean(np.linalg.norm(data[:,:,-1,:],axis=2),axis=1)
np.save('./neural_sde/Echo/50/k_end.npy',end)
print(end.shape)
print(end[-20:])
plt.scatter(np.arange(len(end)),end, s=45, c=end, marker='.',alpha=0.85,cmap='rainbow')
plt.axvline(30,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.9) 
plt.xticks([0,10,20,30,40,50],[20,30,40,50,60,70])
plt.xlabel(r'$k$')
plt.ylabel(r'$\Vert x(0.01)\Vert$')
plot_grid()
plt.colorbar()



plt.subplot(222)
energy_list=np.load('./neural_sde/Echo/50/numerical_energy.npy')
plt.scatter(np.arange(len(energy_list)),energy_list, s=45, c=energy_list, marker='.',alpha=0.85,cmap='rainbow')
plt.xticks([0,10,20],[50,60,70])
plt.xlabel(r'$k$')
plt.ylabel('Energy')
plt.colorbar()
plot_grid()

plt.subplot(223)
time_list=np.load('./neural_sde/Echo/50/numerical_time.npy')
time_list1=np.load('./neural_sde/Echo/50/theory_time.npy')
plt.scatter(np.arange(len(time_list)),time_list, s=45, c='r', marker='.',alpha=0.85,label=r'$\tau_{0.05}~for~k$')
plt.scatter(np.arange(len(time_list1)),time_list1, s=45, c='b', marker='.',alpha=0.85,label=r'$T_{0.05}$')
plt.xticks([0,10,20],[50,60,70])
plt.xlabel(r'$k$')
plt.ylabel('Time')
plt.axhline(0.021285,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.9,label=r'$\tau_{0.05}~for~AS$') 
plot_grid()
plt.legend()

plt.subplot(224)
energy_list=np.log(np.load('./neural_sde/Echo/50/numerical_energy.npy'))
energy_list1=np.log(np.load('./neural_sde/Echo/50/theory_energy.npy'))
plt.scatter(np.arange(len(energy_list)),energy_list, s=45, c='r', marker='.',alpha=0.85,label=r'$\mathcal{E}(\tau_{0.05},T_{0.05})~for~k$')
plt.scatter(np.arange(len(energy_list1)),energy_list1, s=45, c='b', marker='.',alpha=0.85,label=r'$E_{0.05}$')
plt.xticks([0,10,20],[50,60,70])
plt.xlabel(r'$k$')
plt.ylabel('log Energy')
plt.axhline(np.log(877.653),ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.9,label=r'$\mathcal{E}(\tau_{0.05},T_{0.05})~for~AS$') 
plot_grid()
plt.legend()
plt.show()