'''
Created on 20.04.2018

@author: Nils Schmitt
'''
import matplotlib.pyplot as plt
import numpy as np
#from scipy.optimize import curve_fit
#import scipy.integrate as integrate
#from scipy.stats import chi2
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2

####################
#Aufgabe 4.2#
####################
raw = np.loadtxt('data//verstaerkung_4_t62_raw.dat',delimiter = '\t', unpack=True)
a = raw[:1025]
plt.plot(a, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [3]: Coarse Gain 4', fontsize=16)
plt.show()

raw = np.loadtxt('data//verstaerkung_64_t60_raw.dat',delimiter = '\t', unpack=True)
a = raw[0:1025]
plt.plot(a, linestyle='None', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [4]: Coarse Gain 64', fontsize=16)
plt.show()

raw = np.loadtxt('data//verstaerkung_128_t62_raw.dat',delimiter = '\t', unpack=True)
a = raw[0:1025]
plt.plot(a, linestyle='None', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [5]: Coarse Gain 128', fontsize=16)
plt.show()
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
#plt.show()