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
#1 Laser#
####################

######## Leistungs-Strom-Kennlinie ############
I = np.array([1,5,10,15,17.5,20,22.5,25,27.5,30,32.5,35,37.5,40])
I_err = 0.1*np.ones(14)
U_1 = np.array([5.4,5.4,5.9,8.8,11.4,15.6,21.7,30.5,70.5,753,1300,1860,2370,2840])
U_1_err = np.array([0.4,0.4,0.5,0.5,0.5,0.5,0.5,0.7,0.7,1,10,10,10,10])
U_10 = np.array([6.1,8.3,16.8,44.3,74.5,120,192,295,701,4870,5120,5240,5330,5370])
U_10_err = np.array([0.5,0.5,0.5,0.5,0.5,1,1,1,2,10,10,10,10,10])
U_50 = np.array([5.0,5.2,5.2,5.3,5.5,5.7,6.1,6.7,8.3,41.9,71.5,98.4,123,147])
U_50_err = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1])
U_100 = np.array([9.4,30.5,109,365,646,1060,1730,2630,4160,5050,5210,5300,5370,5440])
U_100_err = np.array([0.5,0.5,1,1,1,10,10,10,10,10,10,10,10,10])
P1 = U_1**2/1000
P1_err =2*U_1*U_1_err/1000
P10 = U_10**2/10000
P10_err =2*U_10*U_10_err/10000
P50 = U_50**2/50
P50_err =2*U_50*U_50_err/50
P100 = U_100**2/100000
P100_err =2*U_100*U_100_err/100000

plt.errorbar(I, P1, xerr=I_err, yerr=P1_err,
             fmt='.', linewidth=1,
             linestyle='-', color='black',
             label='Messpunkte mit Fehler für R=1 k$\\Omega$')
plt.errorbar(I, P10, xerr=I_err, yerr=P10_err,
             fmt='.', linewidth=1,
             linestyle='-', color='blue',
             label='Messpunkte mit Fehler für R=10 k$\\Omega$')
plt.errorbar(I, P50, xerr=I_err, yerr=P50_err,
             fmt='.', linewidth=1,
             linestyle='-', color='green',
             label='Messpunkte mit Fehler für R=50 $\\Omega$')
plt.errorbar(I, P100, xerr=I_err, yerr=P100_err,
             fmt='.', linewidth=1,
             linestyle='-', color='magenta',
             label='Messpunkte mit Fehler für R=100 k$\\Omega$')
plt.xlabel('Stromstärke I [mA]', fontsize=13)
plt.ylabel('Leistung [W]', fontsize=13)
plt.title('Abb. [1]: Leistungs-Strom-Kennlinie', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
#plt.savefig('figures//f16_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

############## Strom-Strom-Kennlinie #############
I1 = U_1/1000
I1_err = U_1_err/1000
I10 = U_10/10000
I10_err = U_10_err/10000
I50 = U_50/50
I50_err = U_50_err/50
I100 = U_100/100000
I100_err = U_100_err/100000

def linear(x,a,b):
    return a+x*b

popt1, pcov1 = curve_fit(linear, I[9:], I1[9:], absolute_sigma=True,
                             sigma=I1_err[9:])
a1 = '$a=$'+str(np.round(popt1[1],3))+' $\\pm$ '+str(np.round(np.sqrt(pcov1[1][1]),3))

#popt10, pcov10 = curve_fit(linear, I[8:], I10[8:], absolute_sigma=True,
#                             sigma=I10_err[8:])
#a10 = '$a=$'+str(np.round(popt10[1],3))+' $\\pm$ '+str(np.round(np.sqrt(pcov10[1][1]),3))
popt50, pcov50 = curve_fit(linear, I[9:], I50[9:], absolute_sigma=True,
                             sigma=I50_err[9:])
a50 = '$a=$'+str(np.round(popt50[1],3))+' $\\pm$ '+str(np.round(np.sqrt(pcov50[1][1]),3))
#popt100, pcov100 = curve_fit(linear, I[8:], I100[8:], absolute_sigma=True,
#                             sigma=I100_err[8:])
#a100 = '$a=$'+str(np.round(popt100[1],3))+' $\\pm$ '+str(np.round(np.sqrt(pcov100[1][1]),3))
#chisquare_a0 = np.sum(((linear(I,*popt)-I1)**2/I1_err**2))
#dof = 2
#chisquare_a = chisquare_a0/dof
#print('chi^2_a='+str(chisquare_a))

plt.errorbar(I, I1, xerr=I_err, yerr=I1_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='$R=1 \\mathrm{k}\\Omega$')
plt.errorbar(I, I50, xerr=I_err, yerr=I50_err,
             fmt='.', linewidth=1,
             linestyle='', color='grey',
             label='$R=50 \\Omega$')
plt.errorbar(I, I10, xerr=I_err, yerr=I10_err,
             fmt='.', linewidth=1,
             linestyle='', color='blue',
             label='$R=10 \\mathrm{k}\\Omega$')
plt.errorbar(I, I100, xerr=I_err, yerr=I100_err,
             fmt='.', linewidth=1,
             linestyle='', color='green',
             label='$R=100 \\mathrm{k}\\Omega$')
plt.plot(I[8:],linear(I[8:],*popt1), linestyle='-', marker='',
            color='black', label='linearer Fit $R=1 \\mathrm{k}\\Omega$')
plt.plot(I[8:],linear(I[8:],*popt50), linestyle='-', marker='',
            color='grey', label='linearer Fit $R=50 \\Omega$')
#plt.plot(I[8:],linear(I[8:],*popt10), linestyle='-', marker='',
#            color='blue', label='Messdaten $R=10 \\mathrm{k}\\Omega$')
#plt.plot(I[8:],linear(I[8:],*popt100), linestyle='-', marker='',
#            color='green', label='Messdaten $R=100 \\mathrm{k}\\Omega$')
plt.xlabel('Stromstärke I [mA]', fontsize=13)
plt.ylabel('Stromstärke I$_{PD}$ [mA]', fontsize=13)
plt.title('Abb. [2]: Strom-Strom-Kennlinie', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
#plt.savefig('figures//f16_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

slope1=popt1[1]
slope1_err=np.sqrt(pcov1[1][1])
print('slope1=',slope1,'+/-',slope1_err)
slope50=popt50[1]
slope50_err=np.sqrt(pcov50[1][1])
print('slope50=',slope50,'+/-',slope50_err)


####################
# Transmissionpeak #
####################

def Gauss(x,a,b,c,d):
    return a*np.exp(-(x-b)**2/(2*c**2))+d

x_2, trans_2 = np.loadtxt('data/transmissionpeak.csv', delimiter=',', usecols=(3, 4), unpack=True)
x_4, trans_4 = np.loadtxt('data/transmissionpeak.csv', delimiter=',', usecols=(9, 10), unpack=True)
trans_2=trans_2*100
popt1, pcov1 = curve_fit(Gauss, x_2[91:191], trans_2[91:191],p0=[68,0.0946,0.0003,0.1], absolute_sigma=True)
popt2, pcov2 = curve_fit(Gauss, x_2[1212:1290], trans_2[1212:1290],p0=[68,0.099,0.0003,0.1], absolute_sigma=True)
popt3, pcov3 = curve_fit(Gauss, x_2[2220:2310], trans_2[2220:2310],p0=[68,0.103,0.0003,0.1], absolute_sigma=True)
plt.plot(x_2,trans_2, linestyle='-',
         color='black', label='Messkurve')
plt.plot(x_4,trans_4, linestyle='-',
         color='green', label='Modulation')
plt.plot(x_2[91:191], Gauss(x_2[91:191],*popt1), color='red', label='Gaussian')
plt.plot(x_2[1212:1290], Gauss(x_2[1212:1290],*popt2), color='red', label='Gaussian')
plt.plot(x_2[2220:2310], Gauss(x_2[2220:2310],*popt3), color='red', label='Gaussian')
plt.xlabel('Zeit t [ms]', fontsize=13)
plt.ylabel('Spannung U [mV]', fontsize=13)
plt.title('Abb. [?]: Transmissionpeak', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
#plt.savefig('figures//f16_abb_1.pdf',format='pdf')
print("peak1",popt1)
print("peak2",popt2)
print("peak3",popt3)
#plt.show()
plt.close()


####################
# Modulation #
####################
print('######################')

def Gauss(x,a,b,c,d):
    return a*np.exp(-(x-b)**2/(2*c**2))+d

x_2, trans_2 = np.loadtxt('data/modulation.csv', delimiter=',', usecols=(3, 4), unpack=True)
x_4, trans_4 = np.loadtxt('data/modulation.csv', delimiter=',', usecols=(9, 10), unpack=True)
trans_2=trans_2*100
popt1, pcov1 = curve_fit(Gauss, x_2[76:117], trans_2[76:117],p0=[4,0.094,0.00006,25], absolute_sigma=True)
popt2, pcov2 = curve_fit(Gauss, x_2[124:185], trans_2[124:185],p0=[4,0.0942,0.00006,25], absolute_sigma=True)
popt3, pcov3 = curve_fit(Gauss, x_2[185:238], trans_2[185:238],p0=[4,0.0944,0.00007,25], absolute_sigma=True)
plt.plot(x_2,trans_2, linestyle='-',
         color='black', label='Messkurve')
plt.plot(x_4,trans_4, linestyle='-',
         color='green', label='Modulation')
plt.plot(x_2[76:117], Gauss(x_2[76:117],*popt1), color='red', label='Gaussian')
plt.plot(x_2[124:185], Gauss(x_2[124:185],*popt2), color='red')
plt.plot(x_2[185:238], Gauss(x_2[185:238],*popt3), color='red')
plt.xlabel('Zeit t [ms]', fontsize=13)
plt.ylabel('Spannung U [mV]', fontsize=13)
plt.title('Abb. [?]: Modulation', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
#plt.savefig('figures//f16_abb_1.pdf',format='pdf')
print("linke Modulation",popt1)
print("Mitte",popt2)
print("rechte Modulation",popt3)
plt.show()
plt.close()