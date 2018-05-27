'''
Created on 27.05.2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2


####################
#Aufgabe 1#
####################

f = np.array([1,2,3,4,5,6,7,8,9])
plt.plot(f,f**2, linestyle='None', marker='.',
            color='black', label='Messdaten')
#plt.axis([4e2, 1.5e5, 10, 1.5E3])
plt.xlabel('g(f)', fontsize=13)
plt.ylabel('Frequenz [Hz]', fontsize=13)
plt.title('Abb. [6]: Frequenzgang', fontsize=16)
plt.show()