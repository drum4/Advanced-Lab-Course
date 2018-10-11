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
plt.ylabel('Leistung [$\\mu$W]', fontsize=13)
plt.title('Abb. [1]: Leistungs-Strom-Kennlinie', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_1.pdf',format='pdf')
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
plt.savefig('figures//abb_2.pdf',format='pdf')
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
plt.plot(x_2[91:191], Gauss(x_2[91:191],*popt1), color='red', label='Gauss Fit')
plt.plot(x_2[1212:1290], Gauss(x_2[1212:1290],*popt2), color='red')
plt.plot(x_2[2220:2310], Gauss(x_2[2220:2310],*popt3), color='red')
plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [mV]', fontsize=13)
plt.title('Abb. [3]: Transmissionpeak', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_3.pdf',format='pdf')
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
plt.plot(x_2[76:117], Gauss(x_2[76:117],*popt1), color='red', label='Gauss Fit')
plt.plot(x_2[124:185], Gauss(x_2[124:185],*popt2), color='red')
plt.plot(x_2[185:238], Gauss(x_2[185:238],*popt3), color='red')
plt.plot(x_2[1200:1242], Gauss(x_2[1200:1242],*popt4), color='red')
plt.plot(x_2[1245:1295], Gauss(x_2[1245:1295],*popt5), color='red')
plt.plot(x_2[1295:1346], Gauss(x_2[1295:1346],*popt6), color='red')
plt.plot(x_2[2230:2278], Gauss(x_2[2230:2278],*popt7), color='red')
plt.plot(x_2[2280:2329], Gauss(x_2[2280:2329],*popt8), color='red')
plt.plot(x_2[2330:2370], Gauss(x_2[2330:2370],*popt9), color='red')

plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [mV]', fontsize=13)
plt.title('Abb. [4]: Modulation', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_4.pdf',format='pdf')
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
print('omegaFSR=',omegaFSR,'+',omegaFSR_err,'Freier Spektralbereich')

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
plt.plot(x_2[1236:1265], Gauss(x_2[1236:1265],*popt1), color='red', label='Gauss Fit')
plt.plot(x_2[1330:1355], Gauss(x_2[1330:1355],*popt2), color='red')
plt.plot(x_2[1400:1416], Gauss(x_2[1400:1416],*popt3), color='red')
plt.plot(x_2[1453:1486], Gauss(x_2[1453:1486],*popt4), color='red')
plt.plot(x_2[1536:1556], Gauss(x_2[1536:1556],*popt5), color='red')
plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [mV]', fontsize=13)
plt.title('Abb. [5]: Strom-Frequenz-Charakteristik', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_5.pdf',format='pdf')
print('peak 1',popt1)
print('peak 2',popt2)
print('peak 3',popt3)
print('peak 4',popt4)
print('peak 5',popt5)
#plt.show()
plt.close()

I=27.5
I_err=0.1
t=0.4477
t_err=0.0004
deltaI=I/t
deltaI_err=np.sqrt((I_err/t)**2+(I/t**2*t_err)**2)
print('deltaI=',deltaI,'+',deltaI_err)

diff1=popt2[1]-popt1[1]
diff1_err=np.sqrt(popt2[2]**2+popt1[2]**2)
diff2=popt3[1]-popt2[1]
diff2_err=np.sqrt(popt3[2]**2+popt2[2]**2)
diff3=popt4[1]-popt3[1]
diff3_err=np.sqrt(popt4[2]**2+popt3[2]**2)
diff4=popt5[1]-popt4[1]
diff4_err=np.sqrt(popt5[2]**2+popt4[2]**2)

print("diff1=",diff1,'+',diff1_err)
print("diff2=",diff2,'+',diff2_err)
print("diff3=",diff3,'+',diff3_err)
print("diff4=",diff4,'+',diff4_err)

deltat=(diff1+diff2+diff3+diff4)/4
deltat_err=np.sqrt(diff1_err**2+diff2_err**2+diff3_err**2+diff4_err**2)/4
print("deltat=",deltat,"+",deltat_err)

deltavdeltaI=omegaFSR/(deltaI*deltat)
deltavdeltaI_err=np.sqrt((omegaFSR_err/(deltaI*deltat))**2+(omegaFSR/(deltaI**2*deltat)*deltaI_err)**2+(omegaFSR/(deltaI*deltat**2)*deltat_err)**2)
print('deltavdeltaI=',deltavdeltaI,'+',deltavdeltaI_err,'in Herz pro mA')
#Um wie viel Prozent kann man die Laserfrequenz maximal veraendern#
f=deltavdeltaI*25 #Differenz zwischen 40mA und der Laserschwelle 26.4
f_err=np.sqrt((deltavdeltaI_err*25)**2+(deltavdeltaI*0.1)**2)
print('f=',f,'+',f_err,'In der Abbildung der StromFrequenzCharakteristik sieht man dass die Photodiode bis etwa 15mA anstatt 26.4 detektiert')

######### Frequenz Temperatur Charakteristik #########

x_2, trans_2 = np.loadtxt('data/temperatur.csv', delimiter=',', usecols=(3, 4), unpack=True)
popt1, pcov1 = curve_fit(Gauss, x_2[350:450], trans_2[350:450],p0=[0.3,1.6,0.04,0], absolute_sigma=True)
popt2, pcov2 = curve_fit(Gauss, x_2[576:704], trans_2[576:704],p0=[0.3,2.5,0.04,0], absolute_sigma=True)
popt3, pcov3 = curve_fit(Gauss, x_2[781:881], trans_2[781:881],p0=[0.3,3.3,0.04,0], absolute_sigma=True)
popt4, pcov4 = curve_fit(Gauss, x_2[968:1059], trans_2[968:1059],p0=[0.3,4.0,0.04,0], absolute_sigma=True)
popt5, pcov5 = curve_fit(Gauss, x_2[1134:1225], trans_2[1134:1225],p0=[0.3,4.7,0.04,0], absolute_sigma=True)
popt6, pcov6 = curve_fit(Gauss, x_2[1364:1473], trans_2[1364:1473],p0=[0.3,5.6,0.04,0], absolute_sigma=True)
popt7, pcov7 = curve_fit(Gauss, x_2[1531:1621], trans_2[1531:1621],p0=[0.3,6.3,0.04,0], absolute_sigma=True)
popt8, pcov8 = curve_fit(Gauss, x_2[1698:1789], trans_2[1698:1789],p0=[0.3,6.9,0.04,0], absolute_sigma=True)
popt9, pcov9 = curve_fit(Gauss, x_2[1841:1956], trans_2[1841:1956],p0=[0.3,7.6,0.04,0], absolute_sigma=True)

plt.plot(x_2[85:2240],trans_2[85:2240], linestyle='-',color='black', label='Messkurve')
plt.plot(x_2[350:450], Gauss(x_2[350:450],*popt1), color='red', label='Gaussian')
plt.plot(x_2[576:704], Gauss(x_2[576:704],*popt2), color='red')
plt.plot(x_2[781:881], Gauss(x_2[781:881],*popt3), color='red')
plt.plot(x_2[968:1059], Gauss(x_2[968:1059],*popt4), color='red')
plt.plot(x_2[1134:1225], Gauss(x_2[1134:1225],*popt5), color='red')
plt.plot(x_2[1364:1473], Gauss(x_2[1364:1473],*popt6), color='red')
plt.plot(x_2[1531:1621], Gauss(x_2[1531:1621],*popt7), color='red')
plt.plot(x_2[1698:1789], Gauss(x_2[1698:1789],*popt8), color='red')
plt.plot(x_2[1841:1956], Gauss(x_2[1841:1956],*popt9), color='red')
plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [mV]', fontsize=13)
plt.title('Abb. [6]: Temperatur-Frequenz-Charakteristik', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_6.pdf',format='pdf')
print('peak 1',popt1)
print('peak 2',popt2)
print('peak 3',popt3)
print('peak 4',popt4)
print('peak 5',popt5)
print('peak 6',popt6)
print('peak 7',popt7)
print('peak 8',popt8)
print('peak 9',popt9)
#plt.show()
plt.close()

U=41.2
U_err=0.1
T=U/10*0.39
T_err=U_err/10*0.39
t=7.5
t_err=0.01
deltaT=T/t
deltaT_err=np.sqrt((T_err/t)**2+(T/t**2*t_err)**2)
print('deltaT=',deltaT,'+',deltaT_err)

diff1=popt2[1]-popt1[1]
diff1_err=np.sqrt(popt2[2]**2+popt1[2]**2)
diff2=popt3[1]-popt2[1]
diff2_err=np.sqrt(popt3[2]**2+popt2[2]**2)
diff3=popt4[1]-popt3[1]
diff3_err=np.sqrt(popt4[2]**2+popt3[2]**2)
diff4=popt5[1]-popt4[1]
diff4_err=np.sqrt(popt5[2]**2+popt4[2]**2)
diff5=popt6[1]-popt5[1]
diff5_err=np.sqrt(popt6[2]**2+popt5[2]**2)
diff6=popt7[1]-popt6[1]
diff6_err=np.sqrt(popt7[2]**2+popt6[2]**2)
diff7=popt8[1]-popt7[1]
diff7_err=np.sqrt(popt8[2]**2+popt7[2]**2)
diff8=popt9[1]-popt8[1]
diff8_err=np.sqrt(popt9[2]**2+popt8[2]**2)

print("diff1=",diff1,'+',diff1_err)
print("diff2=",diff2,'+',diff2_err)
print("diff3=",diff3,'+',diff3_err)
print("diff4=",diff4,'+',diff4_err)
print("diff5=",diff5,'+',diff5_err)
print("diff6=",diff6,'+',diff6_err)
print("diff7=",diff7,'+',diff7_err)
print("diff8=",diff8,'+',diff8_err)

deltat=(diff1+diff2+diff3+diff4+diff5+diff6+diff7+diff8)/8
deltat_err=np.sqrt(diff1_err**2+diff2_err**2+diff3_err**2+diff4_err**2+diff5_err**2+diff6_err**2+diff7_err**2+diff8_err**2)/8
print("deltat=",deltat,"+",deltat_err)
a=np.array([diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8])
deltat_err=np.std(a)
print("deltat=",deltat,"+",deltat_err,"statistischer Fehler wird hier genommen")

deltavdeltaT=omegaFSR/(deltaT*deltat)
deltavdeltaT_err=np.sqrt((omegaFSR_err/(deltaT*deltat))**2+(omegaFSR/(deltaT**2*deltat)*deltaT_err)**2+(omegaFSR/(deltaT*deltat**2)*deltat_err)**2)
print('deltavdeltaT=',deltavdeltaT,'+',deltavdeltaT_err)


x_2, trans_2 = np.loadtxt('data/bessel/bessel0_max.csv', delimiter=',', usecols=(3, 4), unpack=True)
plt.plot(x_2[:1851],trans_2[:1851], linestyle='-',color='black', label='Messkurve')

plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [V]', fontsize=13)
plt.title('Abb. [7]: Modulationsindex 0', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_7.pdf',format='pdf')
#plt.show()
plt.close()

x_2, trans_2 = np.loadtxt('data/bessel/bessel1_min.csv', delimiter=',', usecols=(3, 4), unpack=True)
plt.plot(x_2[:1851],trans_2[:1851], linestyle='-',color='black', label='Messkurve')

plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [V]', fontsize=13)
plt.title('Abb. [8]: Modulationsindex 2.41', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_8.pdf',format='pdf')
#plt.show()
plt.close()

x_2, trans_2 = np.loadtxt('data/bessel/bessel2_max.csv', delimiter=',', usecols=(3, 4), unpack=True)
plt.plot(x_2[:1851],trans_2[:1851], linestyle='-',color='black', label='Messkurve')

plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [V]', fontsize=13)
plt.title('Abb. [9]: Modulationsindex 3.83', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_9.pdf',format='pdf')
#plt.show()
plt.close()

x_2, trans_2 = np.loadtxt('data/bessel/bessel3_min.csv', delimiter=',', usecols=(3, 4), unpack=True)
plt.plot(x_2[:1851],trans_2[:1851], linestyle='-',color='black', label='Messkurve')

plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [V]', fontsize=13)
plt.title('Abb. [10]: Modulationsindex 5.52', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_10.pdf',format='pdf')
#plt.show()
plt.close()


##############################
#Dopplerverbreiterte Laserspektroskopie##
#####################################

####### Temperaturscan #######

x_2, trans_2 = np.loadtxt('data/absorption/tempscan_oab.csv', delimiter=',', usecols=(3, 4), unpack=True)
x_3, trans_3 = np.loadtxt('data/absorption/tempscan_oab.csv', delimiter=',', usecols=(9, 10), unpack=True)
x_4, trans_4 = np.loadtxt('data/absorption/tempscan_oab.csv', delimiter=',', usecols=(15, 16), unpack=True)
x_5, trans_5 = np.loadtxt('data/absorption/tempscan_oab.csv', delimiter=',', usecols=(21, 22), unpack=True)
trans_4=trans_4*15
trans_3=trans_3*170

popt1, pcov1 = curve_fit(Gauss, x_3[1226:1249], trans_3[1226:1249],p0=[40,2.47,0.004,4], absolute_sigma=True)
popt2, pcov2 = curve_fit(Gauss, x_3[1286:1302], trans_3[1286:1302],p0=[40,2.59,0.004,4], absolute_sigma=True)
popt3, pcov3 = curve_fit(Gauss, x_3[1344:1360], trans_3[1344:1360],p0=[40,2.70,0.004,4], absolute_sigma=True)
popt4, pcov4 = curve_fit(Gauss, x_3[1400:1412], trans_3[1400:1412],p0=[40,2.81,0.004,4], absolute_sigma=True)
popt5, pcov5 = curve_fit(Gauss, x_3[1456:1470], trans_3[1456:1470],p0=[40,2.92,0.004,4], absolute_sigma=True)
popt6, pcov6 = curve_fit(Gauss, x_3[1507:1522], trans_3[1507:1522],p0=[40,3.02,0.004,4], absolute_sigma=True)
popt7, pcov7 = curve_fit(Gauss, x_3[1561:1574], trans_3[1561:1574],p0=[40,3.13,0.004,4], absolute_sigma=True)
popt8, pcov8 = curve_fit(Gauss, x_3[1613:1625], trans_3[1613:1625],p0=[40,3.24,0.004,4], absolute_sigma=True)
popt9, pcov9 = curve_fit(Gauss, x_3[1665:1676], trans_3[1665:1676],p0=[40,3.34,0.004,4], absolute_sigma=True)
popt10, pcov10 = curve_fit(Gauss, x_3[1716:1728], trans_3[1716:1728],p0=[40,3.44,0.004,4], absolute_sigma=True)
popt11, pcov11 = curve_fit(Gauss, x_4[1211:1281], trans_4[1211:1281],p0=[-40,2.5,0.02,70], absolute_sigma=True)
popt12, pcov12 = curve_fit(Gauss, x_4[1698:1754], trans_4[1698:1754],p0=[-40,3.43,0.02,70], absolute_sigma=True)

#plt.plot(x_2[:],trans_2[:], linestyle='-',color='black', label='Messkurve')
plt.plot(x_3[345:1891],trans_3[345:1891], linestyle='-',color='blue', label='Resonatormessung')
plt.plot(x_4[345:1891],trans_4[345:1891], linestyle='-',color='red', label='Messkurve')
plt.plot(x_5[345:1891],trans_5[345:1891], linestyle='-',color='green', label='Temperaturmodulation')
plt.plot(x_3[1226:1249], Gauss(x_3[1226:1249],*popt1), color='turquoise', label='Gauss Fit')
plt.plot(x_3[1286:1302], Gauss(x_3[1286:1302],*popt2), color='turquoise')
plt.plot(x_3[1344:1360], Gauss(x_3[1344:1360],*popt3), color='turquoise')
plt.plot(x_3[1400:1412], Gauss(x_3[1400:1412],*popt4), color='turquoise')
plt.plot(x_3[1456:1470], Gauss(x_3[1456:1470],*popt5), color='turquoise')
plt.plot(x_3[1507:1522], Gauss(x_3[1507:1522],*popt6), color='turquoise')
plt.plot(x_3[1561:1574], Gauss(x_3[1561:1574],*popt7), color='turquoise')
plt.plot(x_3[1613:1625], Gauss(x_3[1613:1625],*popt8), color='turquoise')
plt.plot(x_3[1665:1676], Gauss(x_3[1665:1676],*popt9), color='turquoise')
plt.plot(x_3[1716:1728], Gauss(x_3[1716:1728],*popt10), color='turquoise')
plt.plot(x_4[1211:1281], Gauss(x_4[1211:1281],*popt11), color='black')
plt.plot(x_4[1698:1754], Gauss(x_4[1698:1754],*popt12), color='black')

plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [V]', fontsize=13)
plt.title('Abb. [11]: Temperaturscan', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_11.pdf',format='pdf')
print('peak 1',popt1)
print('peak 2',popt2)
print('peak 3',popt3)
print('peak 4',popt4)
print('peak 5',popt5)
print('peak 6',popt6)
print('peak 7',popt7)
print('peak 8',popt8)
print('peak 9',popt9)
print('peak 10',popt10)
print('peak 11',popt11)
print('peak 12',popt12)
#plt.show()
plt.close()

diff1=popt2[1]-popt1[1]
diff1_err=np.sqrt(popt2[2]**2+popt1[2]**2)
diff2=popt3[1]-popt2[1]
diff2_err=np.sqrt(popt3[2]**2+popt2[2]**2)
diff3=popt4[1]-popt3[1]
diff3_err=np.sqrt(popt4[2]**2+popt3[2]**2)
diff4=popt5[1]-popt4[1]
diff4_err=np.sqrt(popt5[2]**2+popt4[2]**2)
diff5=popt6[1]-popt5[1]
diff5_err=np.sqrt(popt6[2]**2+popt5[2]**2)
diff6=popt7[1]-popt6[1]
diff6_err=np.sqrt(popt7[2]**2+popt6[2]**2)
diff7=popt8[1]-popt7[1]
diff7_err=np.sqrt(popt8[2]**2+popt7[2]**2)
diff8=popt9[1]-popt8[1]
diff8_err=np.sqrt(popt9[2]**2+popt8[2]**2)
diff9=popt10[1]-popt9[1]
diff9_err=np.sqrt(popt10[2]**2+popt9[2]**2)

print("diff1=",diff1,'+',diff1_err)
print("diff2=",diff2,'+',diff2_err)
print("diff3=",diff3,'+',diff3_err)
print("diff4=",diff4,'+',diff4_err)
print("diff5=",diff5,'+',diff5_err)
print("diff6=",diff6,'+',diff6_err)
print("diff7=",diff7,'+',diff7_err)
print("diff8=",diff8,'+',diff8_err)
print("diff9=",diff9,'+',diff9_err)

deltat=(diff1+diff2+diff3+diff4+diff5+diff6+diff7+diff8+diff9)/9
a=np.array([diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9])
deltat_err=np.std(a)
print("deltat=",deltat,"+",deltat_err,"statistischer Fehler wird hier genommen")

diffsec=popt12[1]-popt11[1]
diffsec_err=np.sqrt(popt12[2]**2+popt11[2]**2)
print('diffsec=',diffsec,'+',diffsec_err)
omegasec=diffsec/deltat*omegaFSR
omegasec_err=np.sqrt((diffsec_err/deltat*omegaFSR)**2+(diffsec/deltat**2*omegaFSR*deltat_err)**2+(diffsec/deltat*omegaFSR_err)**2)
print('omegasec=',omegasec,'+',omegasec_err)


x_2, trans_2 = np.loadtxt('data/absorption/stromscan1_oab.csv', delimiter=',', usecols=(3, 4), unpack=True)
x_3, trans_3 = np.loadtxt('data/absorption/stromscan1_oab.csv', delimiter=',', usecols=(9, 10), unpack=True)
x_4, trans_4 = np.loadtxt('data/absorption/stromscan1_oab.csv', delimiter=',', usecols=(15, 16), unpack=True)
x_5, trans_5 = np.loadtxt('data/absorption/stromscan1_oab.csv', delimiter=',', usecols=(21, 22), unpack=True)

diff_peak1=trans_2-trans_4
trans_3=trans_3*3-0.6
popt1, pcov1 = curve_fit(Gauss, x_2[1328:1459], diff_peak1[1328:1459],p0=[2.5,5.5,0.08,0], absolute_sigma=True)
popt2, pcov2 = curve_fit(Gauss, x_3[1258:1293], trans_3[1258:1293],p0=[0.3,5.1,0.02,-0.55], absolute_sigma=True)
popt3, pcov3 = curve_fit(Gauss, x_3[1341:1360], trans_3[1341:1360],p0=[0.3,5.4,0.02,-0.55], absolute_sigma=True)
popt4, pcov4 = curve_fit(Gauss, x_3[1439:1460], trans_3[1439:1460],p0=[0.3,5.8,0.02,-0.55], absolute_sigma=True)
popt5, pcov5 = curve_fit(Gauss, x_3[1561:1586], trans_3[1561:1586],p0=[0.3,6.3,0.02,-0.55], absolute_sigma=True)

plt.plot(x_3[1051:1741],trans_3[1051:1741], linestyle='-',color='blue', label='Resonatormessung')
plt.plot(x_2[1051:1741],diff_peak1[1051:1741], linestyle='-',color='red', label='Differenz')
#plt.plot(x_5[345:1891],trans_5[345:1891], linestyle='-',color='green', label='Strommodulation')
plt.plot(x_2[1328:1459], Gauss(x_2[1328:1459],*popt1), color='black', label='Gauss Fit')
#plt.plot(x_3[1258:1293], Gauss(x_3[1258:1293],*popt2), color='turquoise')
plt.plot(x_3[1341:1360], Gauss(x_3[1341:1360],*popt3), color='turquoise')
plt.plot(x_3[1439:1460], Gauss(x_3[1439:1460],*popt4), color='turquoise')
#plt.plot(x_3[1561:1586], Gauss(x_3[1561:1586],*popt5), color='turquoise')

plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [V]', fontsize=13)
plt.title('Abb. [12]: F=4 Linie, Differenz', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_12.pdf',format='pdf')
print('peak 1',popt1)
print('resonazpeak 1',popt2)
print('resonazpeak 2',popt3)
print('resonazpeak 3',popt4)
print('resonazpeak 4',popt5)
#plt.show()
plt.close()

diff1=popt3[1]-popt2[1]
diff1_err=np.sqrt(popt3[2]**2+popt2[2]**2)
diff2=popt4[1]-popt3[1]
diff2_err=np.sqrt(popt4[2]**2+popt3[2]**2)
diff3=popt5[1]-popt4[1]
diff3_err=np.sqrt(popt5[2]**2+popt4[2]**2)

#print("diff1=",diff1,'+',diff1_err)
print("diff2=",diff2,'+',diff2_err)
#print("diff3=",diff3,'+',diff3_err)

delhalb=2*np.sqrt(2*np.log(2))*omegaFSR*popt1[2]/diff2
delhalb_err=2*np.sqrt(2*np.log(2))*np.sqrt((omegaFSR_err*popt1[2]/diff2)**2+(omegaFSR*pcov1[2][2]/diff2)**2+(omegaFSR*popt1[2]/diff2**2*diff2_err)**2)
print('delhalb=',delhalb,'+',delhalb_err)


x_2, trans_2 = np.loadtxt('data/absorption/stromscan2_oab.csv', delimiter=',', usecols=(3, 4), unpack=True)
x_3, trans_3 = np.loadtxt('data/absorption/stromscan2_oab.csv', delimiter=',', usecols=(9, 10), unpack=True)
x_4, trans_4 = np.loadtxt('data/absorption/stromscan2_oab.csv', delimiter=',', usecols=(15, 16), unpack=True)
x_5, trans_5 = np.loadtxt('data/absorption/stromscan2_oab.csv', delimiter=',', usecols=(21, 22), unpack=True)

diff_peak2=trans_2-trans_4
trans_3=trans_3*3-0.6
popt1, pcov1 = curve_fit(Gauss, x_2[1434:1547], diff_peak2[1434:1547],p0=[2,5.9,0.08,0.3], absolute_sigma=True)
popt2, pcov2 = curve_fit(Gauss, x_3[1416:1440], trans_3[1416:1440],p0=[0.5,5.7,0.02,-0.55], absolute_sigma=True)
popt3, pcov3 = curve_fit(Gauss, x_3[1516:1540], trans_3[1516:1540],p0=[0.5,6.1,0.02,-0.55], absolute_sigma=True)

plt.plot(x_3[1251:1754],trans_3[1251:1754], linestyle='-',color='blue', label='Resonatormessung')
plt.plot(x_2[1251:1754],diff_peak2[1251:1754], linestyle='-',color='red', label='Differenz')
#plt.plot(x_5[345:1891],trans_5[345:1891], linestyle='-',color='green', label='Strommodulation')
plt.plot(x_2[1434:1547], Gauss(x_2[1434:1547],*popt1), color='black', label='Gauss Fit')
plt.plot(x_3[1416:1440], Gauss(x_3[1416:1440],*popt2), color='turquoise')
plt.plot(x_3[1516:1540], Gauss(x_3[1516:1540],*popt3), color='turquoise')

plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [V]', fontsize=13)
plt.title('Abb. [?]: F=3 Linie, Differenz', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_13.pdf',format='pdf')
print('peak 2',popt1)
print('resonazpeak 1',popt2)
print('resonazpeak 2',popt3)
#plt.show()
plt.close()

diff1=popt3[1]-popt2[1]
diff1_err=np.sqrt(popt3[2]**2+popt2[2]**2)

print("diff1=",diff1,'+',diff1_err)

delhalb=2*np.sqrt(2*np.log(2))*omegaFSR*popt1[2]/diff1
delhalb_err=2*np.sqrt(2*np.log(2))*np.sqrt((omegaFSR_err*popt1[2]/diff1)**2+(omegaFSR*pcov1[2][2]/diff1)**2+(omegaFSR*popt1[2]/diff1**2*diff1_err)**2)
print('delhalb=',delhalb,'+',delhalb_err)


##### Absorbtionsquerschnitt ########

x_2, trans_2 = np.loadtxt('data/absorption/stromscan1_oab.csv', delimiter=',', usecols=(3, 4), unpack=True)
x_3, trans_3 = np.loadtxt('data/absorption/stromscan1_oab.csv', delimiter=',', usecols=(9, 10), unpack=True)
x_4, trans_4 = np.loadtxt('data/absorption/stromscan1_oab.csv', delimiter=',', usecols=(15, 16), unpack=True)

quotient_peak1=trans_4/trans_2
trans_3=trans_3*3-0.6

plt.plot(x_3[1051:1741],trans_3[1051:1741], linestyle='-',color='blue', label='Resonatormessung')
plt.plot(x_2[1051:1741],quotient_peak1[1051:1741], linestyle='-',color='red', label='Quotient')
plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Quotient', fontsize=13)
plt.title('Abb. [14]: F=4 Linie, Quotient', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_14.pdf',format='pdf')
#plt.show()
plt.close()

x_2, trans_2 = np.loadtxt('data/absorption/stromscan2_oab.csv', delimiter=',', usecols=(3, 4), unpack=True)
x_3, trans_3 = np.loadtxt('data/absorption/stromscan2_oab.csv', delimiter=',', usecols=(9, 10), unpack=True)
x_4, trans_4 = np.loadtxt('data/absorption/stromscan2_oab.csv', delimiter=',', usecols=(15, 16), unpack=True)

quotient_peak2=trans_4/trans_2
trans_3=trans_3*3-0.6

plt.plot(x_3[1251:1754],trans_3[1251:1754], linestyle='-',color='blue', label='Resonatormessung')
plt.plot(x_2[1251:1754],quotient_peak2[1251:1754], linestyle='-',color='red', label='Quotient')
plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Quotient', fontsize=13)
plt.title('Abb. [15]: F=3 Linie, Quotient', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_15.pdf',format='pdf')
#plt.show()
plt.close()


#############################################
##### Dopplerfreie Spektroskopie ############
#############################################

x_2, trans_2 = np.loadtxt('data/dopplerfrei/stromscan1.csv', delimiter=',', usecols=(3, 4), unpack=True)
x_3, trans_3 = np.loadtxt('data/dopplerfrei/stromscan1.csv', delimiter=',', usecols=(9, 10), unpack=True)
x_4, trans_4 = np.loadtxt('data/dopplerfrei/stromscan1.csv', delimiter=',', usecols=(15, 16), unpack=True)
x_5, trans_5 = np.loadtxt('data/dopplerfrei/stromscan1.csv', delimiter=',', usecols=(21, 22), unpack=True)

plt.plot(x_3[800:1980],trans_3[800:1980], linestyle='-',color='blue', label='Resonatormessung')
plt.plot(x_4[800:1980],trans_4[800:1980], linestyle='-',color='red', label='Messkurve')
popt1, pcov1 = curve_fit(Gauss, x_3[891:976], trans_3[891:976], p0=[0.2,2.266,0.002,0], absolute_sigma=True)
popt2, pcov2 = curve_fit(Gauss, x_3[1521:1621], trans_3[1521:1621], p0=[0.2,2.33,0.002,0], absolute_sigma=True)

plt.plot(x_3[891:976], Gauss(x_3[891:976],*popt1), color='turquoise', label='Gaussian')
plt.plot(x_3[1521:1621], Gauss(x_3[1521:1621],*popt2), color='turquoise')
plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [V]', fontsize=13)
plt.title('Abb. [16]: Stromscan F=4, dopplerfrei', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_16.pdf',format='pdf')
print('resonazpeak 1',popt1)
print('resonanzpeak 2', popt2)
plt.show()
plt.close()

diff1=popt2[1]-popt1[1]
diff1_err=np.sqrt(popt2[2]**2+popt1[2]**2)

deltat=omegaFSR/diff1
deltat_err=np.sqrt((omegaFSR_err/diff1)**2+(omegaFSR/diff1**2*diff1_err)**2)
print('deltat=',deltat,'+',deltat_err)
diff1=2.32356-2.3068
diff1_err=np.sqrt(0.001**2+0.001**2)
diff2=2.33864-2.32356
diff2_err=np.sqrt(0.002**2+0.001**2)

dreiaufvier=deltat*diff1
dreiaufvier_err=np.sqrt((deltat_err*diff1)**2+(deltat*diff1_err)**2)
vierauffunf=deltat*diff2
vierauffunf_err=np.sqrt((deltat_err*diff2)**2+(deltat*diff2_err)**2)

print('dreiaufvier=',dreiaufvier,'+',dreiaufvier_err)
print('vierauffunf=',vierauffunf,'+',vierauffunf_err)


x_2, trans_2 = np.loadtxt('data/dopplerfrei/stromscan2.csv', delimiter=',', usecols=(3, 4), unpack=True)
x_3, trans_3 = np.loadtxt('data/dopplerfrei/stromscan2.csv', delimiter=',', usecols=(9, 10), unpack=True)
x_4, trans_4 = np.loadtxt('data/dopplerfrei/stromscan2.csv', delimiter=',', usecols=(15, 16), unpack=True)
x_5, trans_5 = np.loadtxt('data/dopplerfrei/stromscan2.csv', delimiter=',', usecols=(21, 22), unpack=True)

plt.plot(x_3[671:1580],trans_3[671:1580], linestyle='-',color='blue', label='Resonatormessung')
plt.plot(x_4[671:1580],trans_4[671:1580], linestyle='-',color='red', label='Messkurve')
popt1, pcov1 = curve_fit(Gauss, x_3[741:831], trans_3[741:831], p0=[0.2,2.3,0.002,0], absolute_sigma=True)
popt2, pcov2 = curve_fit(Gauss, x_3[1428:1511], trans_3[1428:1511], p0=[0.2,2.37,0.002,0], absolute_sigma=True)

plt.plot(x_3[741:831], Gauss(x_3[741:831],*popt1), color='turquoise', label='Gaussian')
plt.plot(x_3[1428:1511], Gauss(x_3[1428:1511],*popt2), color='turquoise')
plt.xlabel('Zeit t [s]', fontsize=13)
plt.ylabel('Spannung U [V]', fontsize=13)
plt.title('Abb. [17]: Stromscan F=3, dopplerfrei', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//abb_17.pdf',format='pdf')
print('resonazpeak 1',popt1)
print('resonanzpeak 2', popt2)
plt.show()
plt.close()


diff1=popt2[1]-popt1[1]
diff1_err=np.sqrt(popt2[2]**2+popt1[2]**2)

deltat=omegaFSR/diff1
deltat_err=np.sqrt((omegaFSR_err/diff1)**2+(omegaFSR/diff1**2*diff1_err)**2)
print('deltat=',deltat,'+',deltat_err)
diff1=2.33739-2.32522
diff1_err=np.sqrt(0.0015**2+0.0007**2)
diff2=2.35015-2.33739
diff2_err=np.sqrt(0.0015**2+0.0007**2)

zweiaufdrei=deltat*diff1
zweiaufdrei_err=np.sqrt((deltat_err*diff1)**2+(deltat*diff1_err)**2)
dreiaufvier=deltat*diff2
dreiaufvier_err=np.sqrt((deltat_err*diff2)**2+(deltat*diff2_err)**2)

print('zweiaufdrei=',zweiaufdrei,'+',zweiaufdrei_err)
print('dreiaufvier=',dreiaufvier,'+',dreiaufvier_err)



