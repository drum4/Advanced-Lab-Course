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

diff1=popt2[1]-popt1[1]
diff1_err=np.sqrt(popt2[2]**2+popt1[2]**2)
diff2=popt3[1]-popt2[1]
diff2_err=np.sqrt(popt3[2]**2+popt2[2]**2)
print("diff1=",diff1,'+',diff1_err)
print("diff2=",diff2,'+',diff2_err)

FSR=(diff1+diff2)/2
FSR_err=np.sqrt(diff1_err**2+diff2_err**2)/2
print('FSR=',FSR,'+',FSR_err)

sigma=np.array([-popt1[2],-popt2[2],popt3[2]])
deltax=np.mean(sigma)*2*np.sqrt(2*np.log(2))
deltax_err=np.std(sigma)*2*np.sqrt(2*np.log(2))
print('deltax=',deltax,'+',deltax_err)


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
popt4, pcov4 = curve_fit(Gauss, x_2[1200:1242], trans_2[1200:1242],p0=[4,0.0984,0.00006,25], absolute_sigma=True)
popt5, pcov5 = curve_fit(Gauss, x_2[1245:1295], trans_2[1245:1295],p0=[4,0.09865,0.00006,25], absolute_sigma=True)
popt6, pcov6 = curve_fit(Gauss, x_2[1295:1346], trans_2[1295:1346],p0=[4,0.09886,0.00007,25], absolute_sigma=True)
popt7, pcov7 = curve_fit(Gauss, x_2[2230:2278], trans_2[2230:2278],p0=[4,0.1026,0.00006,25], absolute_sigma=True)
popt8, pcov8 = curve_fit(Gauss, x_2[2280:2329], trans_2[2280:2329],p0=[4,0.1028,0.00006,25], absolute_sigma=True)
popt9, pcov9 = curve_fit(Gauss, x_2[2330:2370], trans_2[2330:2370],p0=[4,0.10296,0.00007,25], absolute_sigma=True)
plt.plot(x_2,trans_2, linestyle='-',
         color='black', label='Messkurve')
plt.plot(x_4,trans_4, linestyle='-',
         color='green', label='Modulation')
plt.plot(x_2[76:117], Gauss(x_2[76:117],*popt1), color='red', label='Gaussian')
plt.plot(x_2[124:185], Gauss(x_2[124:185],*popt2), color='red')
plt.plot(x_2[185:238], Gauss(x_2[185:238],*popt3), color='red')
plt.plot(x_2[1200:1242], Gauss(x_2[1200:1242],*popt4), color='red')
plt.plot(x_2[1245:1295], Gauss(x_2[1245:1295],*popt5), color='red')
plt.plot(x_2[1295:1346], Gauss(x_2[1295:1346],*popt6), color='red')
plt.plot(x_2[2230:2278], Gauss(x_2[2230:2278],*popt7), color='red')
plt.plot(x_2[2280:2329], Gauss(x_2[2280:2329],*popt8), color='red')
plt.plot(x_2[2330:2370], Gauss(x_2[2330:2370],*popt9), color='red')

plt.xlabel('Zeit t [ms]', fontsize=13)
plt.ylabel('Spannung U [mV]', fontsize=13)
plt.title('Abb. [?]: Modulation', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
#plt.savefig('figures//f16_abb_1.pdf',format='pdf')
print('peak 1')
print("linke Modulation",popt1)
print("Mitte",popt2)
print("rechte Modulation",popt3)
print('peak 2')
print("linke Modulation",popt4)
print("Mitte",popt5)
print("rechte Modulation",popt6)
print('peak 3')
print("linke Modulation",popt7)
print("Mitte",popt8)
print("rechte Modulation",popt9)
#plt.show()
plt.close()

diff1=popt2[1]-popt1[1]
diff1_err=np.sqrt(popt2[2]**2+popt1[2]**2)
diff2=popt3[1]-popt2[1]
diff2_err=np.sqrt(popt3[2]**2+popt2[2]**2)
diff3=popt5[1]-popt4[1]
diff3_err=np.sqrt(popt5[2]**2+popt4[2]**2)
diff4=popt6[1]-popt5[1]
diff4_err=np.sqrt(popt6[2]**2+popt5[2]**2)
diff5=popt8[1]-popt7[1]
diff5_err=np.sqrt(popt8[2]**2+popt7[2]**2)
diff6=popt9[1]-popt8[1]
diff6_err=np.sqrt(popt9[2]**2+popt8[2]**2)

print("diff1=",diff1,'+',diff1_err)
print("diff2=",diff2,'+',diff2_err)
print("diff3=",diff3,'+',diff3_err)
print("diff4=",diff4,'+',diff4_err)
print("diff5=",diff5,'+',diff5_err)
print("diff6=",diff6,'+',diff6_err)

wm=(diff1+diff2+diff3+diff4+diff5+diff6)/6
wm_err=np.sqrt(diff1_err**2+diff2_err**2+diff3_err**2+diff4_err**2+diff5_err**2+diff6_err**2)/6
print("wm=",wm,"+",wm_err,"Messfehler zu gross da fit beim dritten peak nicht passt")
a=np.array([diff1,diff2,diff3,diff4,diff5,diff6])
wm_err=np.std(a)
print("wm=",wm,"+",wm_err,"statistischer Fehler wird hier genommen")

omegaM=56069000
omegaM_err=5000
###Umrechnungsfaktor u###
u=omegaM/wm
u_err=np.sqrt((omegaM_err/wm)**2+(omegaM/wm**2*wm_err)**2)
print('u=',u,'+',u_err)

omegaFSR=u*FSR
omegaFSR_err=np.sqrt((u_err*FSR)**2+(u*FSR_err)**2)
print('omegaFSR=',omegaFSR,'+',omegaFSR_err,'Mittlere Freie Weglaenge')

####Laenge Resonator######
c=299792458
n=1.000292
L=c/(4*omegaFSR*n)
L_err=c/(4*omegaFSR**2*n)*omegaFSR_err
print('L=',L,'+',L_err)

F=FSR/deltax
F_err=np.sqrt((FSR_err/deltax)**2+(FSR/deltax**2*deltax_err)**2)
print('F=',F,'+',F_err)

######Frequenz Strom Charakteristik#########

x_2, trans_2 = np.loadtxt('data/stromfrequenz.csv', delimiter=',', usecols=(3, 4), unpack=True)
popt1, pcov1 = curve_fit(Gauss, x_2[1236:1265], trans_2[1236:1265],p0=[0.5,1.249,0.006,0], absolute_sigma=True)
popt2, pcov2 = curve_fit(Gauss, x_2[1330:1355], trans_2[1330:1355],p0=[0.4,1.34,0.006,0], absolute_sigma=True)
popt3, pcov3 = curve_fit(Gauss, x_2[1400:1416], trans_2[1400:1416],p0=[0.3,1.41,0.006,0], absolute_sigma=True)
popt4, pcov4 = curve_fit(Gauss, x_2[1453:1486], trans_2[1453:1486],p0=[0.2,1.47,0.006,0], absolute_sigma=True)
popt5, pcov5 = curve_fit(Gauss, x_2[1536:1556], trans_2[1536:1556],p0=[0.05,1.547,0.006,0], absolute_sigma=True)


plt.plot(x_2[1142:1610],trans_2[1142:1610], linestyle='-',color='black', label='Messkurve')
plt.plot(x_2[1236:1265], Gauss(x_2[1236:1265],*popt1), color='red', label='Gaussian')
plt.plot(x_2[1330:1355], Gauss(x_2[1330:1355],*popt2), color='red')
plt.plot(x_2[1400:1416], Gauss(x_2[1400:1416],*popt3), color='red')
plt.plot(x_2[1453:1486], Gauss(x_2[1453:1486],*popt4), color='red')
plt.plot(x_2[1536:1556], Gauss(x_2[1536:1556],*popt5), color='red')
plt.xlabel('Zeit t [ms]', fontsize=13)
plt.ylabel('Spannung U [mV]', fontsize=13)
plt.title('Abb. [?]: Strom-Frequenz-Charakteristik', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
#plt.savefig('figures//f16_abb_1.pdf',format='pdf')
print('peak 1',popt1)
print('peak 2',popt2)
print('peak 3',popt3)
print('peak 4',popt4)
print('peak 5',popt5)
plt.show()
plt.close()