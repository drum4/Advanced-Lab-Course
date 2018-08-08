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



##############
#Hystheresis#
##############

current = np.array([0,2,4,6,8,10,12,13])
current_err = 0.1*np.ones(8)

magnetic_up = np.array([[0.002,0.002,0.001],[0.126,0.127,0.126],[0.257,0.268,0.250],[0.373,0.374,0.368],[0.475,0.481,0.474],[0.582,0.589,0.584],[0.668,0.690,0.673],[0.711,0.715,0.710]])
magnetic_up_err = magnetic_up*0.02
mag_up = np.mean(magnetic_up,1)
mag_up_err = np.sqrt(np.mean(magnetic_up_err,1)*1/3)

magnetic_down = np.array([[0.002,0.002,0.002],[0.218,0.125,0.130],[0.254,0.249,0.246],[0.380,0.375,0.374],[0.5,0.490,0.495],[0.590,0.602,0.595],[0.681,0.679,0.674],[0.708,0.713,0.705]])
magnetic_down_err = magnetic_down*0.02
mag_down = np.mean(magnetic_down,1)
mag_down_err = np.sqrt(np.mean(magnetic_down_err,1)*1/3)

def prop(x,a):
    return a*x
popt_up, pcov_up = curve_fit(prop, current, mag_up, absolute_sigma=True,
                             sigma=mag_up_err)
slope_a = '$a_{up}=($'+str(np.round(popt_up[0],3))+' $\\pm$ '+str(np.round(np.sqrt(pcov_up[0][0]),3))+'$)$ mT/A'
chisquare_a = np.sum(((prop(current,*popt_up)-current)**2/current_err**2))
print('chi^2_a='+str(chisquare_a))
popt_down, pcov_down = curve_fit(prop, current, mag_down, absolute_sigma=True,
                                 sigma=mag_down_err)
chisquare_d = np.sum(((prop(current,*popt_down)-current)**2/current_err**2))
print('chi^2_d=',chisquare_d)
slope_d = '$a_{down}=($'+str(np.round(popt_down[0],3))+' $\\pm$ '+str(np.round(np.sqrt(pcov_down[0][0]),3))+'$)$ mT/A'
plt.errorbar(current, mag_up, xerr=current_err, yerr=mag_up_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Ascending measurement')
plt.errorbar(current, mag_down, xerr=current_err, yerr=mag_down_err,
             fmt='.', linewidth=1,
             linestyle='', color='grey',
             label='Descending Measurement')
plt.xlabel('Current I [A]', fontsize=13)
plt.ylabel('Magnetic Field B [mT]', fontsize=13)
plt.title('Fig. [1]: Hysteresis effect for magnets used', fontsize=16)
plt.plot(current, prop(current,*popt_up), color='red', label='Linear fit ascending')
plt.plot(current, prop(current,*popt_down), color='blue', label='Linear fit descending',
         linestyle='--')
plt.text(6,0.05,'Ascending/Descending Slope: \n %s \n %s'%(slope_a, slope_d),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()







#############
#Transversal Measurement#
############

def Gauss(x,a,b,c,d):
    return a*np.exp(-(x-b)**2/(2*c**2))+d


#############
x,cd=np.loadtxt('data/cd.txt', skiprows=1, unpack=True)
popt, pcov = curve_fit(Gauss, x[700:900], cd[700:900],p0=[4*10**6,800,3.94,-1.65*10**7], absolute_sigma=True, maxfev=999999)
plt.plot(x,cd, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data cd', fontsize=16)
plt.plot(x[700:900], Gauss(x[700:900],*popt), color='red', label='Linearer Fit')
print(popt)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

#############

x,ne=np.loadtxt('data/ne.txt', skiprows=1, unpack=True)
popt1, pcov1 = curve_fit(Gauss, x[1:35], ne[1:35],p0=[3*10**6,15,3.5,-1.65*10**7], absolute_sigma=True, maxfev=999999)
popt2, pcov2 = curve_fit(Gauss, x[200:280], ne[200:280],p0=[9*10**6,235,3.5,-1.65*10**7], absolute_sigma=True, maxfev=999999)
popt3, pcov3 = curve_fit(Gauss, x[1040:1170], ne[1040:1170],p0=[16*10**6,1100,3.5,-1.65*10**7], absolute_sigma=True, maxfev=999999)
popt4, pcov4 = curve_fit(Gauss, x[1220:1282], ne[1220:1282],p0=[15*10**6,1265,3.5,-1.65*10**7], absolute_sigma=True, maxfev=999999)
plt.plot(x,ne, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data ne', fontsize=16)
plt.plot(x[1:35], Gauss(x[1:35],*popt1), color='red', label='Linearer Fit')
plt.plot(x[200:280], Gauss(x[200:280],*popt2), color='blue', label='Linearer Fit')
plt.plot(x[1040:1170], Gauss(x[1040:1170],*popt3), color='green', label='Linearer Fit')
plt.plot(x[1220:1282], Gauss(x[1220:1282],*popt4), color='orange', label='Linearer Fit')
print(popt1)
print(popt2)
print(popt3)
print(popt4)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

#############

x,ne_cd=np.loadtxt('data/ne_cd.txt', skiprows=1, unpack=True)
popt5, pcov5 = curve_fit(Gauss, x[700:900], ne_cd[700:900],p0=[6*10**6,800,3.94,-1.65*10**7], absolute_sigma=True, maxfev=999999)
popt6, pcov6 = curve_fit(Gauss, x[1:35], ne_cd[1:35],p0=[3*10**6,15,3.5,-1.65*10**7], absolute_sigma=True, maxfev=999999)
popt7, pcov7 = curve_fit(Gauss, x[200:280], ne_cd[200:280],p0=[10*10**6,235,3.5,-1.65*10**7], absolute_sigma=True, maxfev=999999)
popt8, pcov8 = curve_fit(Gauss, x[1040:1170], ne_cd[1040:1170],p0=[16*10**6,1100,3.5,-1.65*10**7], absolute_sigma=True, maxfev=999999)
popt9, pcov9 = curve_fit(Gauss, x[1220:1279], ne_cd[1220:1279],p0=[15*10**6,1265,3.5,-1.65*10**7], absolute_sigma=True, maxfev=999999)
plt.plot(x,ne_cd, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data ne_cd', fontsize=16)
plt.plot(x[700:900], Gauss(x[700:900],*popt5), color='black', label='Linearer Fit')
plt.plot(x[1:35], Gauss(x[1:35],*popt6), color='red', label='Linearer Fit')
plt.plot(x[200:280], Gauss(x[200:280],*popt7), color='blue', label='Linearer Fit')
plt.plot(x[1040:1170], Gauss(x[1040:1170],*popt8), color='green', label='Linearer Fit')
plt.plot(x[1220:1279], Gauss(x[1220:1279],*popt9), color='orange', label='Linearer Fit')
print('###############################################################')
print(popt5)
print(popt6)
print(popt7)
print(popt8)
print(popt9)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

##############

x,pi_9=np.loadtxt('data/pi_9.txt', skiprows=1, unpack=True)
popt10, pcov10 = curve_fit(Gauss, x[0:60], pi_9[0:60],p0=[3.7*10**6,30,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt11, pcov11 = curve_fit(Gauss, x[80:160], pi_9[80:160],p0=[3.7*10**6,120,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)

plt.plot(x[0:60], Gauss(x[0:60],*popt10), color='blue', label='Linearer Fit')
plt.plot(x[80:160], Gauss(x[80:160],*popt11), color='blue', label='Linearer Fit')

print('###############################################################')
print(popt10)
print(popt11)
plt.plot(x,pi_9, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data pi_9', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
plt.show()
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