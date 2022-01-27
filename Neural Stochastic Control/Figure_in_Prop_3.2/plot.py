import numpy as np
import matplotlib.pyplot as plt
import math
# import pylustrator
# pylustrator.start()

np.random.seed(10)


def nonlinear(N,dt,x0):
    X = []
    X.append(x0)
    z = np.random.normal(0,1,N)
    for i in range(N):
        x = X[i][0]
        new_x = x + x*math.log(abs(x))*dt + 2*x*x*math.sqrt(dt)*z[i]
        X.append([new_x])
    X = np.array(X)
    return X

def linear(k,N,dt,x0):
    X = []
    X.append(x0)
    z = np.random.normal(0,1,N)
    for i in range(N):
        x = X[i][0]
        new_x = x + x*math.log(abs(x))*dt + k*x*math.sqrt(dt)*z[i]
        X.append([new_x])
    X = np.array(X)
    return X



N=200000
dt=0.00001
X1 = linear(1,N,dt,[50.0])
X2 = linear(2,N,dt,[100.0])
X3 = linear(3,N,dt,[150.0])


N=200000
dt=0.000001
Y1 = nonlinear(N,dt,[50.0])
Y2 = nonlinear(N,dt,[100.0])
Y3 = nonlinear(N,dt,[150.0])
fig = plt.figure()

plt1 = fig.add_subplot(121)

plt1.plot(np.arange(N+1),X1,'r',label=r'k=1,$x_0=50.0$')
plt1.plot(np.arange(N+1),X2,'g',label=r'k=2,$x_0=100.0$')
plt1.plot(np.arange(N+1),X3,'b',label=r'k=3,$x_0=150.0$')
plt.legend()
plt2 = fig.add_subplot(122)

plt2.plot(np.arange(N+1),Y1,'r',label=r'$x_0=50.0$')
plt2.plot(np.arange(N+1),Y2,'g',label=r'$x_0=100.0$')
plt2.plot(np.arange(N+1),Y3,'b',label=r'$x_0=150.0$')
plt.legend()



#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(14.710000/2.54, 6.490000/2.54, forward=True)
plt.figure(1).axes[0].set_yscale("symlog")
plt.figure(1).axes[0].set_xlim(-10000.0, 210000.0)
plt.figure(1).axes[0].set_ylim(10.0, 39047767091377.336)
plt.figure(1).axes[0].set_xticks([0.0, 50000.0, 100000.0, 150000.0, 200000.0])
plt.figure(1).axes[0].set_yticks([10.0, 1000.0, 100000.0, 10000000.0, 1000000000.0, 100000000000.0, 10000000000000.0])
plt.figure(1).axes[0].set_xticklabels(["0.0", "0.5", "1.0", "1.5", "2.0"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
plt.figure(1).axes[0].set_yticklabels(["$\mathdefault{10^{1}}$", "$\mathdefault{10^{3}}$", "$\mathdefault{10^{5}}$", "$\mathdefault{10^{7}}$", "$\mathdefault{10^{9}}$", "$\mathdefault{10^{11}}$", "$\mathdefault{10^{13}}$"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="right")
plt.figure(1).axes[0].grid(True)
plt.figure(1).axes[0].legend(frameon=False, borderpad=0.0, labelspacing=0.0, fontsize=7.0, title_fontsize=10.0)
plt.figure(1).axes[0].set_facecolor("#ffffefff")
plt.figure(1).axes[0].set_position([0.097374, 0.228986, 0.368972, 0.647927])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].yaxis.labelpad = -6.320000
plt.figure(1).axes[0].get_legend()._set_loc((0.040311, 0.720466))
plt.figure(1).axes[0].get_legend().set_label("k=1, x(0)=50.0")
plt.figure(1).axes[0].lines[0].set_color("#e96eec")
plt.figure(1).axes[0].lines[0].set_markeredgecolor("#e96eec")
plt.figure(1).axes[0].lines[0].set_markerfacecolor("#e96eec")
plt.figure(1).axes[0].lines[1].set_color("#df4936")
plt.figure(1).axes[0].lines[1].set_markeredgecolor("#df4936")
plt.figure(1).axes[0].lines[1].set_markerfacecolor("#df4936")
plt.figure(1).axes[0].lines[2].set_color("#6ba1ff")
plt.figure(1).axes[0].lines[2].set_markeredgecolor("#6ba1ff")
plt.figure(1).axes[0].lines[2].set_markerfacecolor("#6ba1ff")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("time")
plt.figure(1).axes[0].get_yaxis().get_label().set_fontsize(16)
plt.figure(1).axes[0].get_yaxis().get_label().set_text("x")
plt.figure(1).axes[1].set_xlim(-40.0, 1000.0)
plt.figure(1).axes[1].set_xticks([0.0, 300.0, 600.0, 900.0])
plt.figure(1).axes[1].set_xticklabels(["0.0", "3e-4", "6e-4", "9e-4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
plt.figure(1).axes[1].grid(True)
plt.figure(1).axes[1].legend(frameon=False, borderpad=0.0, labelspacing=0.0, fontsize=7.0, title_fontsize=10.0)
plt.figure(1).axes[1].set_facecolor("#ffffefff")
plt.figure(1).axes[1].set_position([0.563724, 0.228986, 0.368972, 0.647927])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].yaxis.labelpad = -16.967273
plt.figure(1).axes[1].get_legend()._set_loc((0.565661, 0.749353))
plt.figure(1).axes[1].get_legend().set_label("x(0)=50.0")
plt.figure(1).axes[1].lines[0].set_color("#e96eec")
plt.figure(1).axes[1].lines[0].set_markeredgecolor("#e96eec")
plt.figure(1).axes[1].lines[0].set_markerfacecolor("#e96eec")
plt.figure(1).axes[1].lines[1].set_color("#df4936")
plt.figure(1).axes[1].lines[1].set_markeredgecolor("#df4936")
plt.figure(1).axes[1].lines[1].set_markerfacecolor("#df4936")
plt.figure(1).axes[1].lines[2].set_color("#6ba1ff")
plt.figure(1).axes[1].lines[2].set_markeredgecolor("#6ba1ff")
plt.figure(1).axes[1].lines[2].set_markerfacecolor("#6ba1ff")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("time")
plt.figure(1).axes[1].get_yaxis().get_label().set_fontsize(16)
plt.figure(1).axes[1].get_yaxis().get_label().set_text("x")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.256748, 0.935065])
plt.figure(1).texts[0].set_text("(a)")
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[1].new
plt.figure(1).texts[1].set_position([0.718745, 0.935065])
plt.figure(1).texts[1].set_text("(b)")
#% end: automatic generated code from pylustrator
plt.show()
