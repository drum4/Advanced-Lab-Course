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
E_err = np.array([0.00003,0.0003])
M = np.arange(1025)

def linear (x,a):
    return a*x

popt, pcov = curve_fit(linear, ch, E,
                        absolute_sigma=True)
chisquare = np.sum(((linear(ch,*popt)-E)**2/E_err**2))
dof = 1
chisquare_red = chisquare/dof
steigung = 'm ='+str(np.round(popt[0],4))+' $\\pm$ '+str(np.round(np.sqrt(pcov[0][0]),4))+' MeV'
chisquare_text = '$\\chi_{red}^2$ = '+str(np.round(chisquare_red,1))
plt.errorbar(ch, E, xerr=ch_err, fmt=".", linewidth=1,
              linestyle='', color='black',
               label='Messpunkte mit Fehler')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Energie [MeV]', fontsize=13)
plt.title('Abb. [16]: Energie als Funktion der Kanäle',
           fontsize=16)
plt.plot(M, linear(M,*popt),
         color='red', label='Linearer Fit')
plt.text(500,0.5,'%s \n%s'%(steigung,chisquare_text),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_16.pdf',format='pdf')
plt.show()
plt.close()