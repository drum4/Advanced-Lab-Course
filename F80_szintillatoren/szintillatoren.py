'''
Created on 06.05.2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2


################
#Energiekalibrierung#
################

ch = np.array([250,495])
ch_err = np.array([5,10])
E = np.array([0.66166,1.3325])
M = np.linspace(0,1025,5)

def linear (x,a,b):
    return a*x+b

popt, pcov = curve_fit(linear, ch, E,
                        absolute_sigma=True,
                        sigma=ch_err)
plt.errorbar(ch, E, xerr=ch_err, fmt=".", linewidth=1,
              linestyle='', color='black',
               label='Messpunkte mit Fehler')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Energie [MeV]', fontsize=13)
plt.title('Abb. [16]: Energie als Funktion der Kanäle',
           fontsize=16)
plt.plot(ch, linear(M,*popt),
         color='red', label='Linearer Fit')
#plt.text(15000,5,'%s \n%s'%(steigung,chisquare_text),
 #        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
  #        fontsize=13)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_16.pdf',format='pdf')
plt.show()
plt.close()