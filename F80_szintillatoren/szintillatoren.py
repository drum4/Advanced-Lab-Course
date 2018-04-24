'''
Created on 20.04.2018

@author: Nils Schmitt
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2




####################
#Aufgabe 4.2#
####################
raw = np.loadtxt('data//verstaerkung_4_t62_raw.dat', 
                 delimiter = '\t', unpack=True)
a = raw[:1025]
plt.plot(a, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [3]: Coarse Gain 4', fontsize=16)
plt.savefig('figures//f80_abb_3.pdf',format='pdf')
plt.show()

raw = np.loadtxt('data//verstaerkung_64_t60_raw.dat', 
                 delimiter = '\t', unpack=True)
a = raw[0:1025]
plt.plot(a, linestyle='None', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [4]: Coarse Gain 64', fontsize=16)
plt.savefig('figures//f80_abb_4.pdf',format='pdf')
plt.show()

raw = np.loadtxt('data//verstaerkung_128_t62_raw.dat', 
                delimiter = '\t', unpack=True)
a = raw[0:1025]
plt.plot(a, linestyle='None', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [5]: Coarse Gain 128', fontsize=16)
plt.text(850,300,'%s'%('Photopeak'),
         fontsize=13)
plt.arrow(930, 290, 40, -70, shape='full', width=3, 
          length_includes_head=True, color='black')
plt.text(650,120,'%s'%('Compton-Effekt'),
         fontsize=13)
plt.arrow(790, 110, 50, -40, shape='full', width=3, 
          length_includes_head=True, color='black')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_5.pdf',format='pdf')
plt.show()






####################
#Aufgabe 1#
####################
