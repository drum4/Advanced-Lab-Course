'''
Created on 20.04.2018

@author: Nils Schmitt
'''
import matplotlib.pyplot as plt
import numpy as np
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

def emm(x):
    return 11.75+0*x #Verstärkung eintragen
M = np.array([1,50000])

plt.errorbar(f, V, yerr=V_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.yscale('log')
plt.xscale('log')
plt.yticks([1,2,10])
plt.xticks([10,100,1000,10000,5000000])
plt.plot(M,emm(M), linestyle='-', marker='',
            color='red', label='Verstärkung $V_U\\approx 11.75$')
plt.plot(M,emm(M)/np.sqrt(2), linestyle='--', marker='',
            color='red', label='Verstärkung $\\sqrt{2}\\cdot V_U$')
plt.plot(np.array([20.6,20.6]),np.array([5.1,11]), linestyle=':', marker='',
            color='blue', label='untere Grenzfrequenz $f=20.6$ Hz')
plt.xlabel('Frequenz [Hz]', fontsize=13)
plt.ylabel('Verstärkung', fontsize=13)
plt.title('Abb. [2]: Frequenzgang der Emitterschaltung', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//e08_abb_2.pdf',format='pdf')
#plt.show()
plt.close()




####################
#Kollektorschaltung#
####################

f = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,500,1000,5000,10000,50000,100000,1000000,10000000])
U_ein = np.array([742,741,740,724,723,724,725,725,725,722,722,724,726,725,720,719,720,717,723,725,723,714,637])
U_ein_err = np.array([6,5,5,3,3,3,4,3,3,3,4,3,3,3,3,3,3,2,3,4,4,2,2])
U_aus = np.array([182,333,438,514,564,602,623,641,655,664,673,680,686,688,690,711,715,710,716,722,718,705,664])
U_aus_err = np.array([2,2,3,3,2,2,2,3,3,3,2,3,3,3,3,3,3,2,5,4,5,3,7])

V = U_aus/U_ein
V_err = np.sqrt((U_aus_err/U_ein)**2+(U_aus*U_ein_err/U_ein**2)**2)

def koll(x):
    return 1+0*x #Verstärkung eintragen
M = np.array([1,50000])

plt.errorbar(f, V, yerr=V_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.yscale('log')
plt.xscale('log')
plt.xticks([10,100,1000,10000,5000000])
plt.plot(M,koll(M), linestyle='-', marker='',
            color='red', label='Verstärkung $V_U\\approx 1$')
plt.plot(M,koll(M)/np.sqrt(2), linestyle='--', marker='',
            color='red', label='Verstärkung $\\sqrt{2}\\cdot V_U$')
plt.plot(np.array([41.8,41.8]),np.array([0.3,0.95]), linestyle=':', marker='',
            color='blue', label='untere Grenzfrequenz $f=41.8$ Hz')
plt.xlabel('Frequenz [Hz]', fontsize=13)
plt.ylabel('Verstärkung', fontsize=13)
plt.title('Abb. [6]: Frequenzgang der Kollektorschaltung', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//e08_abb_6.pdf',format='pdf')
#plt.show()
plt.close()