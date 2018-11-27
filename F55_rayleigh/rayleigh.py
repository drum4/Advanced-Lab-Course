'''
Created on Nov 27, 2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.optimize import curve_fit

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2









##############
#Teil 1#
##############

x_2, trans_2 = np.loadtxt('data/transmissionpeak.csv', delimiter=',', usecols=(3, 4), unpack=True)

plt.errorbar(current, mag_up, xerr=current_err, yerr=mag_up_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Ascending measurement')
plt.xlabel('Current I [A]', fontsize=13)
plt.ylabel('Magnetic Field B [mT]', fontsize=13)
plt.title('Fig. [1]: Hysteresis effect for magnets used', fontsize=16)
plt.plot(current, prop(current,*popt_up), color='red', label='Linear fit ascending')
plt.plot(current, prop(current,*popt_down), color='blue', label='Linear fit descending',
         linestyle='--')
plt.text(5,0.05,'Ascending/Descending Slope: \n %s \n %s'%(slope_a, slope_d),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()