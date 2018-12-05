'''
Created on Dec 5, 2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2


def gaus(x, t):
    return 1/(2*np.sqrt(np.pi*t))*np.exp(-x*x/(4*t))

plt.xlabel('Ort x', fontsize=13)
plt.ylabel('Konzentration c', fontsize=13)
#plt.title('Fig. [1]: Gd500 relaxationtime $T_1$', fontsize=16)
t = np.arange(-50,50,.2)
plt.plot(t, gaus(t,5), color='black', label='t = 5')
plt.plot(t, gaus(t,15), color=(.784,0,0), label='t = 15')
plt.plot(t, gaus(t,50), color='forestgreen', label='t = 50')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//gaus.pdf',format='pdf')
#plt.show()
plt.close()
