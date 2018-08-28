'''
Created on Aug 16, 2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2





#################
#Relaxation Time#
#################
def lattice(x,a,b,c):
    return a*(1-2*np.exp(-x/b))+c

x, y, y_err=np.loadtxt('data/relaxation/Ga500_T1.txt', skiprows=0, unpack=True)

popt, pcov = curve_fit(lattice, x, y, absolute_sigma=True,
                        p0=[10.8,190,1],sigma=y_err)

plt.errorbar(x, y, yerr=y_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Measuring data')
plt.xlabel('Time [ms]', fontsize=13)
plt.ylabel('Amplitude', fontsize=13)
#plt.title('Fig. [1]: Gd500 Relaxationtime $T_1$', fontsize=16)
t = np.arange(20,750)
plt.plot(t, lattice(t,*popt), color='red', label='Exponential fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_1.png',format='png')
#plt.show()
plt.close()

################ Spin-Echo ################
def spin(x,a,b,c):
    return a*np.exp(-x/b)+c

x, y, y_err=np.loadtxt('data/relaxation/Ga500_T2.txt', skiprows=0, unpack=True)

popt, pcov = curve_fit(lattice, x, y, absolute_sigma=True,
                        p0=[10.9,154,3],sigma=y_err)

plt.errorbar(x, y, yerr=y_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Measuring data')
plt.xlabel('Time [ms]', fontsize=13)
plt.ylabel('Amplitude', fontsize=13)
#plt.title('Fig. [2]: Gd500 Relaxationtime $T_2$', fontsize=16)
t = np.arange(20,380)
plt.plot(t, lattice(t,*popt), color='red', label='Exponential fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_2.png',format='png')
#plt.show()
plt.close()

############ Car-Purcell ###########
x, y, y_err=np.loadtxt('data/relaxation/Ga500_CP.txt', skiprows=0, unpack=True)

popt, pcov = curve_fit(lattice, x, y, absolute_sigma=True,
                        p0=[86,170,3],sigma=y_err)

plt.errorbar(x, y, yerr=y_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Measuring data')
plt.xlabel('Time [ms]', fontsize=13)
plt.ylabel('Amplitude', fontsize=13)
#plt.title('Fig. [3]: Gd500 Relaxationtime $T_2$ Car-Purcell', fontsize=16)
t = np.arange(20,700)
plt.plot(t, lattice(t,*popt), color='red', label='Exponential fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_3.png',format='png')
#plt.show()
plt.close()









##########################
#Chemical Shift#
##########################
x, y=np.loadtxt('data/samples/Bplus.txt', skiprows=0, unpack=True)

plt.plot(x[51:], y[51:], color='black', label='Measuring Data')
plt.xlabel('Frequency [Hz]', fontsize=13)
plt.ylabel('Amplitude', fontsize=13)
#plt.title('Fig. [4]: Sample B+', fontsize=16)
plt.annotate("TMS-Peak", xy=(350, 0.105), xytext=(100, 0.115), fontsize=13,
            arrowprops=dict(arrowstyle="simple", color='black'))
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_4.png',format='png')
#plt.show()
plt.close()









#########################
#Oil#
#########################
x1, y1=np.loadtxt('data/testProfil1.txt', skiprows=0, unpack=True)
x2, y2=np.loadtxt('data/testProfil2.txt', skiprows=0, unpack=True)
x3, y3=np.loadtxt('data/testProfil3.txt', skiprows=0, unpack=True)

plt.plot(x1, y1, color='black', label='Less Oil')
plt.plot(x2, y2, color='forestgreen', label='Much Oil')
plt.plot(x3, y3, color='darkblue', label='Oil and Teflon')
plt.xlabel('Vertical Coordinates [mm]', fontsize=13)
plt.ylabel('Amplitude', fontsize=13)
#plt.title('Fig. [5]: Different Oil', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_5.png',format='png')
#plt.show()
plt.close()









########################
#Sand#
########################
x1, y1=np.loadtxt('data/oilinsand/testProfil538976318.txt', skiprows=0, unpack=True)
x2, y2=np.loadtxt('data/oilinsand/testProfil538976323.txt', skiprows=0, unpack=True)
x3, y3=np.loadtxt('data/oilinsand/testProfil538976326.txt', skiprows=0, unpack=True)
x4, y4=np.loadtxt('data/oilinsand/testProfil538976333.txt', skiprows=0, unpack=True)
x5, y5=np.loadtxt('data/oilinsand/testProfil538976336.txt', skiprows=0, unpack=True)
x6, y6=np.loadtxt('data/oilinsand/testProfil538976343.txt', skiprows=0, unpack=True)
x7, y7=np.loadtxt('data/oilinsand/testProfil538976346.txt', skiprows=0, unpack=True)
x8, y8=np.loadtxt('data/oilinsand/testProfil538976353.txt', skiprows=0, unpack=True)
x9, y9=np.loadtxt('data/oilinsand/testProfil538976356.txt', skiprows=0, unpack=True)
x10, y10=np.loadtxt('data/oilinsand/testProfil538976376.txt', skiprows=0, unpack=True)
x11, y11=np.loadtxt('data/oilinsand/testProfil538976386.txt', skiprows=0, unpack=True)
x12, y12=np.loadtxt('data/oilinsand/testProfil538976396.txt', skiprows=0, unpack=True)

plt.plot(x1[299:], y1[299:], color='black', label='Measured Data')
plt.plot(x2[299:], y2[299:], color='forestgreen')
plt.plot(x3[299:], y3[299:], color='darkblue')
plt.plot(x4[299:], y4[299:])
plt.plot(x5[299:], y5[299:])
plt.plot(x6[299:], y6[299:])
plt.plot(x7[299:], y7[299:])
plt.plot(x8[299:], y8[299:])
plt.plot(x9[299:], y9[299:])
plt.plot(x10[299:], y10[299:])
plt.plot(x11[299:], y11[299:])
plt.plot(x12[299:], y12[299:])
plt.xlabel('Vertical Coordinates [mm]', fontsize=13)
plt.ylabel('Amplitude', fontsize=13)
#plt.title('Fig. [6]: Oil in Sand', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_6.png',format='png')
#plt.show()
plt.close()