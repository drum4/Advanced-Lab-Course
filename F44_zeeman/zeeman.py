'''
Created on 07.08.2018

@author: Nils Schmitt
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.optimize import curve_fit

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2


def Gauss(x,a,b,c,d):
    return a*np.exp(-(x-b)**2/c)+d


#############
x,cd=np.loadtxt('data/cd.txt', skiprows=1, unpack=True)
popt, pcov = curve_fit(Gauss, x[:-1], cd[:-1],p0=[1,800,1,-1.65*10**-10], absolute_sigma=True)
plt.plot(x,cd, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data cd', fontsize=16)
plt.plot(x, Gauss(x,*popt), color='red', label='Linearer Fit')
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
plt.show()
plt.close()

#############

x,ne=np.loadtxt('data/ne.txt', skiprows=1, unpack=True)
plt.plot(x,ne, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data ne', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

#############

x,ne_cd=np.loadtxt('data/ne_cd.txt', skiprows=1, unpack=True)
plt.plot(x,ne_cd, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data ne_cd', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

##############

x,pi_9=np.loadtxt('data/pi_9.txt', skiprows=1, unpack=True)
plt.plot(x,pi_9, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data pi_9', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

##############

x,pi_11=np.loadtxt('data/pi_11.txt', skiprows=1, unpack=True)
plt.plot(x,pi_11, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data pi_11', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

##############

x,pi_13=np.loadtxt('data/pi_13.txt', skiprows=1, unpack=True)
plt.plot(x,pi_13, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data pi_13', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

##############

x,sigma_9=np.loadtxt('data/sigma_9.txt', skiprows=1, unpack=True)
plt.plot(x,sigma_9, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data sigma_9', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

##############

x,sigma_11=np.loadtxt('data/sigma_11.txt', skiprows=1, unpack=True)
plt.plot(x,sigma_11, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data sigma_11', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

##############

x,sigma_13=np.loadtxt('data/sigma_13.txt', skiprows=1, unpack=True)
plt.plot(x,sigma_13, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data sigma_13', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()