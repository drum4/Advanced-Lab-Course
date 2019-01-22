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





#######################
### Relaxation Time ###
#######################
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
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [1]: Gd500 relaxationtime $T_1$', fontsize=16)
t = np.arange(20,750)
plt.plot(t, lattice(t,*popt), color=(.784,0,0), label='Exponential fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

x, y, y_err=np.loadtxt('data/relaxation/Ga600_T1.txt', skiprows=0, unpack=True)

popt, pcov = curve_fit(lattice, x, y, absolute_sigma=True,
                        p0=[10.8,190,1],sigma=y_err)

plt.errorbar(x, y, yerr=y_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Measuring data')
plt.xlabel('Time [ms]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [1]: Gd600 relaxationtime $T_1$', fontsize=16)
t = np.arange(20,750)
plt.plot(t, lattice(t,*popt), color=(.784,0,0), label='Exponential fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_1_600.pdf',format='pdf')
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
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [2]: Gd500 Relaxationtime $T_2$', fontsize=16)
t = np.arange(20,380)
plt.plot(t, lattice(t,*popt), color=(.784,0,0), label='Exponential fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_2.pdf',format='pdf')
#plt.show()
plt.close()

x, y, y_err=np.loadtxt('data/relaxation/Ga600_T2.txt', skiprows=0, unpack=True)

popt, pcov = curve_fit(lattice, x, y, absolute_sigma=True,
                        p0=[10.9,154,3],sigma=y_err)

plt.errorbar(x, y, yerr=y_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Measuring data')
plt.xlabel('Time [ms]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [2]: Gd600 Relaxationtime $T_2$', fontsize=16)
t = np.arange(20,380)
plt.plot(t, lattice(t,*popt), color=(.784,0,0), label='Exponential fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_2_600.pdf',format='pdf')
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
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [3]: Gd500 Relaxationtime $T_2$ Car-Purcell', fontsize=16)
t = np.arange(20,700)
plt.plot(t, lattice(t,*popt), color=(.784,0,0), label='Exponential fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_3.pdf',format='pdf')
#plt.show()
plt.close()

x, y, y_err=np.loadtxt('data/relaxation/Ga600_CP.txt', skiprows=0, unpack=True)

popt, pcov = curve_fit(lattice, x, y, absolute_sigma=True,
                        p0=[86,170,3],sigma=y_err)

plt.errorbar(x, y, yerr=y_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Measuring data')
plt.xlabel('Time [ms]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [3]: Gd600 Relaxationtime $T_2$ Car-Purcell', fontsize=16)
t = np.arange(20,700)
plt.plot(t, lattice(t,*popt), color=(.784,0,0), label='Exponential fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_3_600.pdf',format='pdf')
#plt.show()
plt.close()







##########################
#Chemical Shift#
##########################
x, y=np.loadtxt('data/samples/A+.txt', skiprows=0, unpack=True)

plt.plot(x[110:295], y[110:295], color='black', label='Measuring data')
plt.xlabel('Frequency [Hz]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [4]: Sample A+', fontsize=16)
plt.annotate("TMS-Peak", xy=(325, 1.4), xytext=(250, 3.1), fontsize=13,
            arrowprops=dict(arrowstyle="simple", color='black'))
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_4.pdf',format='pdf')
#plt.show()
plt.close()

x, y=np.loadtxt('data/samples/Bplus.txt', skiprows=0, unpack=True)

plt.plot(400,0.14, color='black')
plt.plot(x[77:350], y[77:350], color='black', label='Measuring data')
plt.xlabel('Frequency [Hz]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [5]: Sample B+', fontsize=16)
plt.annotate("TMS-Peak", xy=(357, 0.1085), xytext=(250, 0.13), fontsize=13,
            arrowprops=dict(arrowstyle="simple", color='black'))
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_5.pdf',format='pdf')
#plt.show()
plt.close()

x, y=np.loadtxt('data/samples/Cplus.txt', skiprows=0, unpack=True)

plt.plot(x[120:450], y[120:450], color='black', label='Measuring data')
plt.xlabel('Frequency [Hz]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [6]: Sample C+', fontsize=16)
plt.annotate("TMS-Peak", xy=(650, 0.1), xytext=(720, 0.14), fontsize=13,
            arrowprops=dict(arrowstyle="simple", color='black'))
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_6.pdf',format='pdf')
#plt.show()
plt.close()

x, y=np.loadtxt('data/samples/C.txt', skiprows=1, unpack=True)

plt.plot(x[120:450], y[120:450], color='black', label='Measuring data')
plt.xlabel('Frequency [Hz]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [6]: Sample C+', fontsize=16)
#plt.annotate("TMS-Peak", xy=(650, 0.1), xytext=(720, 0.14), fontsize=13,
 #           arrowprops=dict(arrowstyle="simple", color='black'))
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_6_ohne.pdf',format='pdf')
#plt.show()
plt.close()

x, y=np.loadtxt('data/samples/D+.txt', skiprows=0, unpack=True)

plt.plot(x[247:450], y[247:450], color='black', label='Measuring data')
plt.xlabel('Frequency [Hz]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [7]: Sample D+', fontsize=16)
plt.annotate("TMS-Peak", xy=(603, 2.7), xytext=(540, 4.5), fontsize=13,
            arrowprops=dict(arrowstyle="simple", color='black'))
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_7.pdf',format='pdf')
#plt.show()
plt.close()

x, y=np.loadtxt('data/samples/D.txt', skiprows=1, unpack=True)

plt.plot(x[247:450], y[247:450], color='black', label='Measuring data')
plt.xlabel('Frequency [Hz]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [7_1]: Sample D', fontsize=16)
#plt.annotate("TMS-Peak", xy=(603, 2.7), xytext=(540, 4.5), fontsize=13,
 #           arrowprops=dict(arrowstyle="simple", color='black'))
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_7_ohne.pdf',format='pdf')
#plt.show()
plt.close()










#########################
#Oil#
#########################
x1, y1=np.loadtxt('data/testProfil1.txt', skiprows=0, unpack=True)
x2, y2=np.loadtxt('data/testProfil2.txt', skiprows=0, unpack=True)
x3, y3=np.loadtxt('data/testProfil3.txt', skiprows=0, unpack=True)

plt.plot(-20,280)
plt.plot(x1, y1, color=(.784,0,0), label='Less oil')
plt.plot(x2, y2, color='forestgreen', label='Much oil')
#plt.plot(x3, y3, color='black', label='Oil between teflon layers')
plt.xlabel('Vertical coordinates [mm]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [8]: Different oil', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_8.pdf',format='pdf')
#plt.show()
plt.close()


plt.plot(x3, y3, color='black', label='Oil between teflon layers')
plt.xlabel('Vertical coordinates [mm]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [8]: Different oil', fontsize=16)
plt.legend(loc = 'upper right', frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_8_teflon.pdf',format='pdf')
#plt.show()
plt.close()









########################
#Sand# Zur Übersichtlichkeit im Plot nicht alle benutzen 
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

plt.plot(x1[299:], y1[299:], color='black', label='$t=0$ s')
plt.plot(x2[299:], y2[299:], color=(.784,0,0), label='$t=50$ s')
#plt.plot(x3[299:], y3[299:])
#plt.plot(x4[299:], y4[299:])
#plt.plot(x5[299:], y5[299:])
#plt.plot(x6[299:], y6[299:])
#plt.plot(x7[299:], y7[299:])
plt.plot(x8[299:], y8[299:], color='forestgreen', label='$t=350$ s')
#plt.plot(x9[299:], y9[299:])
plt.plot(x10[299:], y10[299:], color=(0,.3,.6), label='$t=580$ s')
#plt.plot(x11[299:], y11[299:])
#plt.plot(x12[299:], y12[299:])
plt.xlabel('Vertical coordinates [mm]', fontsize=13)
plt.ylabel('Amplitude [a. u.]', fontsize=13)
#plt.title('Fig. [9]: Oil in sand', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f61_abb_9.pdf',format='pdf')
#plt.show()
plt.close()