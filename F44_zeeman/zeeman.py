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
plt.title('Fig. [2]: Cadmium Spectrum', fontsize=16)
plt.plot(x[700:900], Gauss(x[700:900],*popt), color='red', label='Linearer Fit')
print("cd:",popt)
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
plt.title('Fig. [3]: Neon spectrum', fontsize=16)
plt.plot(x[1:35], Gauss(x[1:35],*popt1), color='red', label='Linearer Fit')
plt.plot(x[200:280], Gauss(x[200:280],*popt2), color='blue', label='Linearer Fit')
plt.plot(x[1040:1170], Gauss(x[1040:1170],*popt3), color='green', label='Linearer Fit')
plt.plot(x[1220:1282], Gauss(x[1220:1282],*popt4), color='orange', label='Linearer Fit')
print("ne_1:",popt1)
print("ne_2:",popt2)
print("ne_3:",popt3)
print("ne_4:",popt4)
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
plt.title('Fig. [4]: Neon and Cadmium Spectrum', fontsize=16)
plt.plot(x[700:900], Gauss(x[700:900],*popt5), color='black', label='Linearer Fit')
plt.plot(x[1:35], Gauss(x[1:35],*popt6), color='red', label='Linearer Fit')
plt.plot(x[200:280], Gauss(x[200:280],*popt7), color='blue', label='Linearer Fit')
plt.plot(x[1040:1170], Gauss(x[1040:1170],*popt8), color='green', label='Linearer Fit')
plt.plot(x[1220:1279], Gauss(x[1220:1279],*popt9), color='orange', label='Linearer Fit')
print('########################### Cd, Ne 1 bis 4 ##############################')
print(popt5)
print(popt6)
print(popt7)
print(popt8)
print(popt9)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

####################
#Wavelength Cadmium#
####################

pos = np.array([popt6[1],popt7[1],popt8[1],popt9[1]])
pos_err = np.array([popt6[2],popt7[2],popt8[2],popt9[2]])
w = np.array([653.29,650.65,640.22,638.30])

def linear(x,a,b):
    return a+b*x

popt_wave, pcov_wave = curve_fit(linear, w, pos, absolute_sigma=True,
                             sigma=pos_err)
a = '$a = $('+str(np.round(popt_wave[0],1))+' $\\pm$ '+str(np.round(np.sqrt(pcov_wave[0,0]),1))+') px'
b = '$b = $('+str(np.round(popt_wave[1],1))+' $\\pm$ '+str(np.round(np.sqrt(pcov_wave[1,1]),1))+') px/nm'
plt.errorbar(w, pos, yerr=pos_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Measuring data')
plt.xlabel('Wavelength [nm]', fontsize=13)
plt.ylabel('Position [px]', fontsize=13)
plt.title('Fig. [?]: Wave to ', fontsize=16)
plt.plot(w, linear(w,*popt_wave), color='red', label='Linear fit')
plt.text(639,100,'Linear fit: \n %s \n %s'%(a, b),
         bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
         fontsize=13)
plt.legend(frameon=True, fontsize = 12)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

wave_cd = (popt5[1]-popt_wave[0])/popt_wave[1]
wave_cd_err = np.sqrt((popt5[2]/popt_wave[1])**2+(np.sqrt(pcov_wave[0,0])/popt_wave[1])**2+((popt5[1]-popt_wave[0])/popt_wave[1]**2*np.sqrt(pcov_wave[1,1]))**2)
print("###########################################")
print("Wavelength Cadmium")
print("lambda_cd = ",wave_cd," + ",wave_cd_err)



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

print('################### pi_9 Peak 1 bis 12 ############################')
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

def polynom(x,b0,b1,b2):
    return b0+x*b1+x**2*b2

a=np.array([popt10[1],popt11[1],popt12[1],popt13[1],popt14[1],popt15[1],popt16[1],popt17[1],popt18[1],popt19[1],popt20[1],popt21[1]])
a_err=np.array([popt10[2],popt11[2],popt12[2],popt13[2],popt14[2],popt15[2],popt16[2],popt17[2],popt18[2],popt19[2],popt20[2],popt21[2]])
k=np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

popt_pi_9, pcov_pi_9 = curve_fit(polynom, a, k, absolute_sigma=True)
plt.errorbar(a, k, xerr=a_err, fmt='.')
plt.plot(a, polynom(a,*popt_pi_9), linestyle='-', color='red', label='quadratic plot')
plt.xlabel('Position of the lines [px]', fontsize=13)
plt.ylabel('Order', fontsize=13)
plt.title('Abb. [??]: Orders of the pi-Lines at 9 A', fontsize=16)
plt.text(500,8.5,'Fitfunction b0+b1*x+b2*x**2 \n b0= %s \n b1= %s \n b2= %s'%(popt_pi_9[0],popt_pi_9[1],popt_pi_9[2]),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=11)
plt.legend(frameon=True, fontsize = 12)
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

print('################# pi_11 Peak 1 bis 12 ############################')
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


a=np.array([popt22[1],popt23[1],popt24[1],popt25[1],popt26[1],popt27[1],popt28[1],popt29[1],popt30[1],popt31[1],popt32[1],popt33[1]])
a_err=np.array([popt22[2],popt23[2],popt24[2],popt25[2],popt26[2],popt27[2],popt28[2],popt29[2],popt30[2],popt31[2],popt32[2],popt33[2]])
k=np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

popt_pi_11, pcov_pi_11 = curve_fit(polynom, a, k, absolute_sigma=True)
plt.errorbar(a, k, xerr=a_err, fmt='.')
plt.plot(a, polynom(a,*popt_pi_11), linestyle='-', color='red', label='quadratic plot')
plt.xlabel('Position of the lines [px]', fontsize=13)
plt.ylabel('Order', fontsize=13)
plt.title('Abb. [??]: Orders of the pi-Lines at 11 A', fontsize=16)
plt.text(500,8.5,'Fitfunction b0+b1*x+b2*x**2 \n b0= %s \n b1= %s \n b2= %s'%(popt_pi_11[0],popt_pi_11[1],popt_pi_11[2]),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=11)
plt.legend(frameon=True, fontsize = 12)
#plt.show()
plt.close()


##############

x,pi_13=np.loadtxt('data/pi_13.txt', skiprows=1, unpack=True)

popt34, pcov34 = curve_fit(Gauss, x[10:50], pi_13[10:50],p0=[3*10**6,30,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt35, pcov35 = curve_fit(Gauss, x[100:140], pi_13[100:140],p0=[3*10**6,120,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt36, pcov36 = curve_fit(Gauss, x[193:230], pi_13[193:230],p0=[3*10**6,210,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt37, pcov37 = curve_fit(Gauss, x[290:330], pi_13[290:330],p0=[3*10**6,308,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt38, pcov38 = curve_fit(Gauss, x[385:430], pi_13[385:430],p0=[3*10**6,408,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt39, pcov39 = curve_fit(Gauss, x[490:535], pi_13[490:535],p0=[3*10**6,510,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt40, pcov40 = curve_fit(Gauss, x[600:640], pi_13[600:640],p0=[3*10**6,620,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt41, pcov41 = curve_fit(Gauss, x[710:755], pi_13[710:755],p0=[3*10**6,730,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt42, pcov42 = curve_fit(Gauss, x[825:875], pi_13[825:875],p0=[3*10**6,850,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt43, pcov43 = curve_fit(Gauss, x[950:1000], pi_13[950:1000],p0=[3*10**6,970,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt44, pcov44 = curve_fit(Gauss, x[1085:1130], pi_13[1085:1130],p0=[3*10**6,1105,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)
popt45, pcov45 = curve_fit(Gauss, x[1225:1270], pi_13[1225:1270],p0=[3*10**6,1250,8,-1.37*10**7], absolute_sigma=True, maxfev=999999)

plt.plot(x[10:50], Gauss(x[10:50],*popt34), color='blue', label='Linearer Fit')
plt.plot(x[100:140], Gauss(x[100:140],*popt35), color='blue', label='Linearer Fit')
plt.plot(x[193:230], Gauss(x[193:230],*popt36), color='blue', label='Linearer Fit')
plt.plot(x[290:330], Gauss(x[290:330],*popt37), color='blue', label='Linearer Fit')
plt.plot(x[385:430], Gauss(x[385:430],*popt38), color='blue', label='Linearer Fit')
plt.plot(x[490:535], Gauss(x[490:535],*popt39), color='blue', label='Linearer Fit')
plt.plot(x[600:640], Gauss(x[600:640],*popt40), color='blue', label='Linearer Fit')
plt.plot(x[710:755], Gauss(x[710:755],*popt41), color='blue', label='Linearer Fit')
plt.plot(x[825:875], Gauss(x[825:875],*popt42), color='blue', label='Linearer Fit')
plt.plot(x[950:1000], Gauss(x[950:1000],*popt43), color='blue', label='Linearer Fit')
plt.plot(x[1085:1130], Gauss(x[1085:1130],*popt44), color='blue', label='Linearer Fit')
plt.plot(x[1225:1270], Gauss(x[1225:1270],*popt45), color='blue', label='Linearer Fit')

print('################### pi_13 Peak 1 bis 12 ###############################')
print(popt34)
print(popt35)
print(popt36)
print(popt37)
print(popt38)
print(popt39)
print(popt40)
print(popt41)
print(popt42)
print(popt43)
print(popt44)
print(popt45)

plt.plot(x,pi_13, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data pi_13', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

a=np.array([popt34[1],popt35[1],popt36[1],popt37[1],popt38[1],popt39[1],popt40[1],popt41[1],popt42[1],popt43[1],popt44[1],popt45[1]])
a_err=np.array([popt34[2],popt35[2],popt36[2],popt37[2],popt38[2],popt39[2],popt40[2],popt41[2],popt42[2],popt43[2],popt44[2],popt45[2]])
k=np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

popt_pi_13, pcov_pi_13 = curve_fit(polynom, a, k, absolute_sigma=True)
plt.errorbar(a, k, xerr=a_err, fmt='.')
plt.plot(a, polynom(a,*popt_pi_13), linestyle='-', color='red', label='quadratic plot')
plt.xlabel('Position of the lines [px]', fontsize=13)
plt.ylabel('Order', fontsize=13)
plt.title('Abb. [??]: Orders of the pi-Lines at 13 A', fontsize=16)
plt.text(500,8.5,'Fitfunction b0+b1*x+b2*x**2 \n b0= %s \n b1= %s \n b2= %s'%(popt_pi_13[0],popt_pi_13[1],popt_pi_13[2]),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=11)
plt.legend(frameon=True, fontsize = 12)
#plt.show()
plt.close()




##############

x,sigma_9=np.loadtxt('data/sigma_9.txt', skiprows=1, unpack=True)

popt46, pcov46 = curve_fit(Gauss, x[0:37], sigma_9[0:37],p0=[1*10**6,14,8,-1.36*10**7], absolute_sigma=True, maxfev=999999)
popt47, pcov47 = curve_fit(Gauss, x[38:70], sigma_9[38:70],p0=[1*10**6,55,8,-1.36*10**7], absolute_sigma=True, maxfev=999999)
popt48, pcov48 = curve_fit(Gauss, x[87:115], sigma_9[87:115],p0=[1*10**6,102,8,-1.36*10**7], absolute_sigma=True, maxfev=999999)

plt.plot(x[0:37], Gauss(x[0:37],*popt46), color='blue', label='Linearer Fit')
plt.plot(x[38:70], Gauss(x[38:70],*popt47), color='blue', label='Linearer Fit')
plt.plot(x[87:115], Gauss(x[87:115],*popt48), color='blue', label='Linearer Fit')

print('################### sigma_9 Peak 1 bis 24 ###############################')
print(popt46)
print(popt47)
print(popt48)


plt.plot(x,sigma_9, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Pixel', fontsize=13)
plt.ylabel('Intensity', fontsize=13)
plt.title('Abb. [??]: Data sigma_9', fontsize=16)
#plt.savefig('figures//f44_abb_1.pdf',format='pdf')
plt.show()
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