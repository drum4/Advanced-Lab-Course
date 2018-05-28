'''
Created on 20.04.2018

@author: Nils Schmitt
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2


####################
#Emitterschaltung#
####################

f = np.array([10,20,30,40,50,60,70,80,90,100,110,200,500,1000,5000,10000,50000,10000,500000])
U_ein = np.array([520,520,520,520,519,519,520,520,520,520,520,519,518,519,519,520,522,520,515])*10**(-3)
U_ein_err = np.array([2,2,2,2,2,3,1,2,2,2,2,3,3,2,3,2,3,3,2])*10**(-3)
U_aus = np.array([2.88,5.10,5.36,5.83,6.18,6.39,6.62,6.65,6.90,6.80,6.79,7.10,6.30,6.12,5.80,5.87,5.62,5.05,1.51])
U_aus_err = np.array([0.01,0.02,0.02,0.03,0.03,0.05,0.04,0.05,0.04,0.04,0.04,0.08,0.11,0.22,0.12,0.11,0.09,0.07,0.03])

V = U_aus/U_ein
V_err = np.sqrt((U_aus_err/U_ein)**2+(U_aus*U_ein_err/U_ein**2)**2)

fg = np.ones(2)*11.75
M = np.array([1,50000])

plt.errorbar(f, V, yerr=V_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.yscale('log')
plt.xscale('log')
plt.yticks([1,2,10])
plt.xticks([10,100,1000,10000,5000000])
plt.plot(M,fg, linestyle='-', marker='',
            color='red', label='Grenzfrequenz')
plt.xlabel('Frequenz [Hz]', fontsize=13)
plt.ylabel('Verstärkung', fontsize=13)
plt.title('Abb. [1]: Frequenzgang der Emitterschaltung', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//e08_abb_1.pdf',format='pdf')
plt.show()
plt.close()




####################
#Emitterschaltung#
####################