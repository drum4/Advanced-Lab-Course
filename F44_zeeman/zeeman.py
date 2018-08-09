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
popt10, pcov10 = curve_fit(Gauss, x[0:60], pi_9[0:60],p0=[3*10**6,30,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt11, pcov11 = curve_fit(Gauss, x[80:160], pi_9[80:160],p0=[3*10**6,120,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt12, pcov12 = curve_fit(Gauss, x[190:230], pi_9[190:230],p0=[3*10**6,210,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt13, pcov13 = curve_fit(Gauss, x[280:340], pi_9[280:340],p0=[3*10**6,308,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt14, pcov14 = curve_fit(Gauss, x[390:430], pi_9[390:430],p0=[3*10**6,408,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt15, pcov15 = curve_fit(Gauss, x[495:530], pi_9[495:530],p0=[3*10**6,510,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt16, pcov16 = curve_fit(Gauss, x[600:640], pi_9[600:640],p0=[3*10**6,620,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt17, pcov17 = curve_fit(Gauss, x[710:750], pi_9[710:750],p0=[3*10**6,730,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt18, pcov18 = curve_fit(Gauss, x[830:870], pi_9[830:870],p0=[3*10**6,850,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt19, pcov19 = curve_fit(Gauss, x[950:1000], pi_9[950:1000],p0=[3*10**6,970,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt20, pcov20 = curve_fit(Gauss, x[1085:1125], pi_9[1085:1125],p0=[3*10**6,1105,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt21, pcov21 = curve_fit(Gauss, x[1230:1265], pi_9[1230:1265],p0=[3*10**6,1250,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)

plt.plot(x[0:60], Gauss(x[0:60],*popt10), color='blue', label='Linearer Fit')
plt.plot(x[80:160], Gauss(x[80:160],*popt11), color='blue', label='Linearer Fit')
plt.plot(x[190:230], Gauss(x[190:230],*popt12), color='blue', label='Linearer Fit')
plt.plot(x[280:340], Gauss(x[280:340],*popt13), color='blue', label='Linearer Fit')
plt.plot(x[390:430], Gauss(x[390:430],*popt14), color='blue', label='Linearer Fit')
plt.plot(x[495:530], Gauss(x[495:530],*popt15), color='blue', label='Linearer Fit')
plt.plot(x[600:640], Gauss(x[600:640],*popt16), color='blue', label='Linearer Fit')
plt.plot(x[710:750], Gauss(x[710:750],*popt17), color='blue', label='Linearer Fit')
plt.plot(x[830:870], Gauss(x[830:870],*popt18), color='blue', label='Linearer Fit')
plt.plot(x[950:1000], Gauss(x[950:1000],*popt19), color='blue', label='Linearer Fit')
plt.plot(x[1085:1125], Gauss(x[1085:1125],*popt20), color='blue', label='Linearer Fit')
plt.plot(x[1230:1265], Gauss(x[1230:1265],*popt21), color='blue', label='Linearer Fit')

print('###############################################################')
print(popt10)
print(popt11)
print(popt12)
print(popt13)
print(popt14)
print(popt15)
print(popt16)
print(popt17)
print(popt18)
print(popt19)
print(popt20)
print(popt21)
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

popt22, pcov22 = curve_fit(Gauss, x[15:50], pi_11[15:50],p0=[3*10**6,30,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt23, pcov23 = curve_fit(Gauss, x[105:150], pi_11[105:150],p0=[3*10**6,120,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt24, pcov24 = curve_fit(Gauss, x[200:240], pi_11[200:240],p0=[3*10**6,210,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt25, pcov25 = curve_fit(Gauss, x[290:340], pi_11[290:340],p0=[3*10**6,308,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt26, pcov26 = curve_fit(Gauss, x[390:430], pi_11[390:430],p0=[3*10**6,408,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt27, pcov27 = curve_fit(Gauss, x[495:540], pi_11[495:540],p0=[3*10**6,510,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt28, pcov28 = curve_fit(Gauss, x[605:645], pi_11[605:645],p0=[3*10**6,620,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt29, pcov29 = curve_fit(Gauss, x[710:755], pi_11[710:755],p0=[3*10**6,730,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt30, pcov30 = curve_fit(Gauss, x[830:875], pi_11[830:875],p0=[3*10**6,850,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt31, pcov31 = curve_fit(Gauss, x[955:1000], pi_11[955:1000],p0=[3*10**6,970,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt32, pcov32 = curve_fit(Gauss, x[1085:1130], pi_11[1085:1130],p0=[3*10**6,1105,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt33, pcov33 = curve_fit(Gauss, x[1230:1270], pi_11[1230:1270],p0=[3*10**6,1250,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)

plt.plot(x[15:50], Gauss(x[15:50],*popt22), color='blue', label='Linearer Fit')
plt.plot(x[105:150], Gauss(x[105:150],*popt23), color='blue', label='Linearer Fit')
plt.plot(x[200:240], Gauss(x[200:240],*popt24), color='blue', label='Linearer Fit')
plt.plot(x[290:340], Gauss(x[290:340],*popt25), color='blue', label='Linearer Fit')
plt.plot(x[390:430], Gauss(x[390:430],*popt26), color='blue', label='Linearer Fit')
plt.plot(x[495:540], Gauss(x[495:540],*popt27), color='blue', label='Linearer Fit')
plt.plot(x[605:645], Gauss(x[605:645],*popt28), color='blue', label='Linearer Fit')
plt.plot(x[710:755], Gauss(x[710:755],*popt29), color='blue', label='Linearer Fit')
plt.plot(x[830:875], Gauss(x[830:875],*popt30), color='blue', label='Linearer Fit')
plt.plot(x[955:1000], Gauss(x[955:1000],*popt31), color='blue', label='Linearer Fit')
plt.plot(x[1085:1130], Gauss(x[1085:1130],*popt32), color='blue', label='Linearer Fit')
plt.plot(x[1230:1270], Gauss(x[1230:1270],*popt33), color='blue', label='Linearer Fit')

print('###############################################################')
print(popt22)
print(popt23)
print(popt24)
print(popt25)
print(popt26)
print(popt27)
print(popt28)
print(popt29)
print(popt30)
print(popt31)
print(popt32)
print(popt33)

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