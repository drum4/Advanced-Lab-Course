'''
Created on Sep 29, 2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2





#################
#Saugvermögen#
#################
PA = 1013
vol1 = 0.04*np.ones(5)*10**(-3)
vol1_err = np.sqrt(0.0003**2+0.001**2)*10**(-3)
t1 = np.array([122.8,57.1,24.7,12.6,8.7])
t1_err = np.array([0.3,0.1,0.1,0.2,0.3])
p1 = np.array([5.1e-6,1e-5,2.5e-5,5e-5,7.5e-5])
p1_err = np.array([0.1e-6,0.1e-5,0.1e-5,0.1e-5,0.1e-5])
kap1 = vol1/t1*PA/p1
kap1_err = vol1/t1**2*PA/p1*t1_err

vol2 = 0.1*np.ones(3)*10**(-3)
vol2_err = np.sqrt(0.0006**2+0.002**2)*10**(-3)
t2 = np.array([20.6,15.6,6.2])
t2_err = np.array([0.1,0.2,0.3])
p2 = np.array([7.5e-5,1e-4,2.5e-4])
p2_err = np.array([0.1e-5,0.1e-4,0.1e-4])
kap2 = vol2/t2*PA/p2
kap2_err = vol2/t2**2*PA/p2*t2_err

plt.errorbar(p1, kap1, xerr=p1_err, yerr=kap1_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Measuring data')
plt.errorbar(p2, kap2, xerr=p2_err, yerr=kap2_err,
             fmt='.', linewidth=1,
             linestyle='', color='red',
             label='Measuring data')
plt.xscale('log')
plt.show()





##################
#Leitwerte Rohr#
##################
p_ur = np.array([9.7e-6,1e-5,1.1e-5,1.5e-5,2.4e-5,3.3e-5,4.1e-5,9.2e-5,1.9e-4,3.4e-4,6.6e-4,2.9e-3,9.8e-3,2.5e-2,7.4e-2])
p_ur_err = np.array([.1e-6,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-4,.1e-4,.1e-4,.1e-3,.1e-3,.1e-2,.1e-2])
p_or = np.array([4.9e-4,7.5e-4,1.1e-3,2.5e-3,4.8e-3,7.6e-3,1e-2,2.5e-2,5e-2,7.6e-2,.1,.25,.5,.76,1.1])
p_or_err = np.array([.1e-4,.1e-4,.1e-3,.1e-3,.1e-3,.1e-3,.1e-2,.1e-2,.1e-2,.1e-2,.1e-1,.1e-1,.1e-1,.1e-1,.1])










##################
#Leitwerte Blende#
##################
p_ob = np.array([8.5e-5,9.6e-5,2.4e-4,5.7e-4,7.8e-4,1.1e-3,2.4e-3,5.1e-3,7.6e-3,1e-2,2.6e-2,5.2e-2,7.5e-2,9.7e-2,.25,.5,.76])