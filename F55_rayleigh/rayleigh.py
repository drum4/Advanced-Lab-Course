'''
Created on Nov 27, 2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.optimize import curve_fit
from matplotlib.dates import datestr2num

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2

Delta_2H, Delta_2H_StDev, Delta_18O, Delta_18O_StDev, Delta_17O, Delta_17O_StDev= np.loadtxt('data/teil1_40_50-Processed.txt', usecols=(2,3,4,5,6,7), skiprows=1, unpack=True)


def auslesen(a,b):
    x=np.array([[],[],[],[],[],[]])
    for i in [a,b]:
         x=np.insert(x,0,[[Delta_2H[i], Delta_2H_StDev[i], Delta_18O[i], Delta_18O_StDev[i], Delta_17O[i], Delta_17O_StDev[i]]],axis=1)
    return x

def auslesenstandard(a,b,c,d,e,f,g):
    x=np.array([[],[],[],[],[],[]])
    for i in [a,b,c,d,e,f,g]:
         x=np.insert(x,0,[[Delta_2H[i], Delta_2H_StDev[i], Delta_18O[i], Delta_18O_StDev[i], Delta_17O[i], Delta_17O_StDev[i]]],axis=1)
    return x

def meansample(x):
    x_mean=np.array([])
    x_mean=np.append(x_mean, [(x[0][0]+x[0][1])/2])
    x_mean=np.append(x_mean, [np.sqrt((x[1][0])**2+(x[1][1])**2)/2])
    x_mean=np.append(x_mean, [(x[2][0]+x[2][1])/2])
    x_mean=np.append(x_mean, [np.sqrt((x[3][0])**2+(x[3][1])**2)/2])
    x_mean=np.append(x_mean, [(x[4][0]+x[4][1])/2])
    x_mean=np.append(x_mean, [np.sqrt((x[5][0])**2+(x[5][1])**2)/2])
    return x_mean



Degre40_1252=auslesen(6,38)#tatsaeliche Zeilennummer aus txt Datei -2
Degre40_1323=auslesen(8,40)
Degre40_1352=auslesen(10,42)
Degre40_1423=auslesen(18,50)
Degre40_1453=auslesen(20,52)
Degre40_1521=auslesen(28,60)
Degre40_1552=auslesen(30,62)

Degre40_1252_mean=meansample(Degre40_1252)
Degre40_1323_mean=meansample(Degre40_1323)
Degre40_1352_mean=meansample(Degre40_1352)
Degre40_1423_mean=meansample(Degre40_1423)
Degre40_1453_mean=meansample(Degre40_1453)
Degre40_1521_mean=meansample(Degre40_1521)
Degre40_1552_mean=meansample(Degre40_1552)


delta_D_40_1=np.array([Degre40_1252[0][0],Degre40_1323[0][0],Degre40_1352[0][0],Degre40_1423[0][0],Degre40_1453[0][0],Degre40_1521[0][0],Degre40_1552[0][0]])
delta_D_40_2=np.array([Degre40_1252[0][1],Degre40_1323[0][1],Degre40_1352[0][1],Degre40_1423[0][1],Degre40_1453[0][1],Degre40_1521[0][1],Degre40_1552[0][1]])
delta_D_err_40_1=np.array([Degre40_1252[1][0],Degre40_1323[1][0],Degre40_1352[1][0],Degre40_1423[1][0],Degre40_1453[1][0],Degre40_1521[1][0],Degre40_1552[1][0]])
delta_D_err_40_2=np.array([Degre40_1252[1][1],Degre40_1323[1][1],Degre40_1352[1][1],Degre40_1423[1][1],Degre40_1453[1][1],Degre40_1521[1][1],Degre40_1552[1][1]])
delta_O18_40_1=np.array([Degre40_1252[2][0],Degre40_1323[2][0],Degre40_1352[2][0],Degre40_1423[2][0],Degre40_1453[2][0],Degre40_1521[2][0],Degre40_1552[2][0]])
delta_O18_40_2=np.array([Degre40_1252[2][1],Degre40_1323[2][1],Degre40_1352[2][1],Degre40_1423[2][1],Degre40_1453[2][1],Degre40_1521[2][1],Degre40_1552[2][1]])
delta_O18_err_40_1=np.array([Degre40_1252[3][0],Degre40_1323[3][0],Degre40_1352[3][0],Degre40_1423[3][0],Degre40_1453[3][0],Degre40_1521[3][0],Degre40_1552[3][0]])
delta_O18_err_40_2=np.array([Degre40_1252[3][1],Degre40_1323[3][1],Degre40_1352[3][1],Degre40_1423[3][1],Degre40_1453[3][1],Degre40_1521[3][1],Degre40_1552[3][1]])
delta_O17_40=np.array([Degre40_1252[4],Degre40_1323[4],Degre40_1352[4],Degre40_1423[4],Degre40_1453[4],Degre40_1521[4],Degre40_1552[4]])
delta_O17_err_40=np.array([Degre40_1252[5],Degre40_1323[5],Degre40_1352[5],Degre40_1423[5],Degre40_1453[5],Degre40_1521[5],Degre40_1552[5]])

Masse_40=np.array([1038.20,1026.02,1014.70,1003.19,991.83,981.25,969.81])
fraction_40=np.array([Masse_40[0]/Masse_40[0],Masse_40[1]/Masse_40[0],Masse_40[2]/Masse_40[0],Masse_40[3]/Masse_40[0],Masse_40[4]/Masse_40[0],Masse_40[5]/Masse_40[0]])

Degre50_1252=auslesen(7,39)
Degre50_1324=auslesen(9,41)
Degre50_1353=auslesen(17,49)
Degre50_1424=auslesen(19,51)
Degre50_1455=auslesen(21,53)
Degre50_1522=auslesen(29,61)
Degre50_1553=auslesen(31,63)

Degre50_1252_mean=meansample(Degre50_1252)
Degre50_1324_mean=meansample(Degre50_1324)
Degre50_1353_mean=meansample(Degre50_1353)
Degre50_1424_mean=meansample(Degre50_1424)
Degre50_1455_mean=meansample(Degre50_1455)
Degre50_1522_mean=meansample(Degre50_1522)
Degre50_1553_mean=meansample(Degre50_1553)

Masse_50=np.array([1012.61,990.05,968.50,946.43,925.30,903.97,891.91])
fraction_50=Masse_50/Masse_50[0]
#
#
VE_1=auslesenstandard(0,15,27,32,47,59,68)
Alpen_1=auslesenstandard(2,13,23,34,45,55,66)
Colle_1=auslesenstandard(4,11,26,36,43,58,64)
Sammelprobe_1=auslesenstandard(5,16,25,37,48,57,69)


Delta_2H, Delta_2H_StDev, Delta_18O, Delta_18O_StDev, Delta_17O, Delta_17O_StDev= np.loadtxt('data/teil1_60_teil3_4-Processed.txt', usecols=(2,3,4,5,6,7), skiprows=1, unpack=True)

Degre60_1321=auslesen(20,53)
Degre60_1351=auslesen(21,54)
Degre60_1428=auslesen(28,61)
Degre60_1455=auslesen(29,62)
Degre60_1525=auslesen(30,63)
Degre60_1554=auslesen(31,64)
Degre60_1623=auslesen(32,65)

Degre60_1321_mean=meansample(Degre60_1321)
Degre60_1351_mean=meansample(Degre60_1351)
Degre60_1428_mean=meansample(Degre60_1428)
Degre60_1455_mean=meansample(Degre60_1455)
Degre60_1525_mean=meansample(Degre60_1525)
Degre60_1554_mean=meansample(Degre60_1554)
Degre60_1623_mean=meansample(Degre60_1623)


Masse_60=np.array([953.10,919.13,875.74,844.71,809.80,776.12,740.07])
fraction_60=np.array([Masse_60[0]/Masse_60[0],Masse_60[1]/Masse_60[0],Masse_60[2]/Masse_60[0],Masse_60[3]/Masse_60[0],Masse_60[4]/Masse_60[0],Masse_60[5]/Masse_60[0]])

Sample_293=auslesen(6,39)
Sample_58=auslesen(7,40)
Sample_298=auslesen(8,41)
Sample_k009=auslesen(9,42)
Sample_w051=auslesen(10,43)
Sample_w075=auslesen(17,50)
Sample_w023=auslesen(18,51)
Sample_w077=auslesen(19,52)

VE_2=auslesenstandard(0,15,26,33,48,59,70)
Alpen_2=auslesenstandard(2,13,23,35,45,57,68)
Colle_2=auslesenstandard(4,16,27,37,49,55,66)
Sammelprobe_2=auslesenstandard(5,11,25,38,47,60,71)

temp, humidity = np.loadtxt('data/temp_data_logger.txt', delimiter="\t", usecols=(2,3), skiprows=1, unpack=True)

temp_experiment1=np.mean(temp[155:194])
temp_experiment1_err=np.std(temp[155:194])
humidity_experiment1=np.mean(humidity[155:194])
humidity_experiment1_err=np.std(humidity[155:194])
print('temp_experiment1=',temp_experiment1,'+',temp_experiment1_err)
print('humidity_experiment1=',humidity_experiment1,'+',humidity_experiment1_err)

temp_experiment2=np.mean(temp[160:189])
temp_experiment2_err=np.std(temp[160:189])
humidity_experiment2=np.mean(humidity[160:189])
humidity_experiment2_err=np.std(humidity[160:189])
print('temp_experiment2=',temp_experiment2,'+',temp_experiment2_err)
print('humidity_experiment2=',humidity_experiment2,'+',humidity_experiment2_err)

temp_experiment3=np.mean(temp[451:489])
temp_experiment3_err=np.std(temp[451:489])
humidity_experiment3=np.mean(humidity[451:489])
humidity_experiment3_err=np.std(humidity[451:489])
print('temp_experiment3=',temp_experiment3,'+',temp_experiment3_err)
print('humidity_experiment3=',humidity_experiment3,'+',humidity_experiment3_err)


a=np.arange(1,8)
b=np.arange(9,16)
plt.errorbar(a, VE_1[0], yerr=VE_1[1], fmt='.', linewidth=1, linestyle='', color='black', label='VE')
plt.errorbar(a, Colle_1[0], yerr=Colle_1[1], fmt='.', linewidth=1, linestyle='', color='green', label='Colle')
plt.errorbar(a, Alpen_1[0], yerr=Alpen_1[1], fmt='.', linewidth=1, linestyle='', color='turquoise', label='Alpen')
plt.errorbar(a, Sammelprobe_1[0], yerr=Sammelprobe_1[1], fmt='.', linewidth=1, linestyle='', color='blue', label='Sammelprobe')
plt.errorbar(b, VE_2[0], yerr=VE_2[1], fmt='.', linewidth=1, linestyle='', color='black')
plt.errorbar(b, Colle_2[0], yerr=Colle_2[1], fmt='.', linewidth=1, linestyle='', color='green')
plt.errorbar(b, Alpen_2[0], yerr=Alpen_2[1], fmt='.', linewidth=1, linestyle='', color='turquoise')
plt.errorbar(b, Sammelprobe_2[0], yerr=Sammelprobe_2[1], fmt='.', linewidth=1, linestyle='', color='blue')
plt.xlabel('Nummer', fontsize=13)
plt.ylabel('Delta_2H', fontsize=13)
plt.title('Fig. [1]:Delta_2H Standards', fontsize=16)
plt.legend()
#plt.show()
plt.close()

plt.errorbar(a, VE_1[2], yerr=VE_1[3], fmt='.', linewidth=1, linestyle='', color='black', label='VE')
plt.errorbar(a, Colle_1[2], yerr=Colle_1[3], fmt='.', linewidth=1, linestyle='', color='green', label='Colle')
plt.errorbar(a, Alpen_1[2], yerr=Alpen_1[3], fmt='.', linewidth=1, linestyle='', color='turquoise', label='Alpen')
plt.errorbar(a, Sammelprobe_1[2], yerr=Sammelprobe_1[3], fmt='.', linewidth=1, linestyle='', color='blue', label='Sammelprobe')
plt.errorbar(b, VE_2[2], yerr=VE_2[3], fmt='.', linewidth=1, linestyle='', color='black')
plt.errorbar(b, Colle_2[2], yerr=Colle_2[3], fmt='.', linewidth=1, linestyle='', color='green')
plt.errorbar(b, Alpen_2[2], yerr=Alpen_2[3], fmt='.', linewidth=1, linestyle='', color='turquoise')
plt.errorbar(b, Sammelprobe_2[2], yerr=Sammelprobe_2[3], fmt='.', linewidth=1, linestyle='', color='blue')
plt.xlabel('Nummer', fontsize=13)
plt.ylabel('Delta_18O', fontsize=13)
plt.title('Fig. [1]:Delta_18O Standards', fontsize=16)
plt.legend()
#plt.show()
plt.close()

plt.errorbar(a, VE_1[4], yerr=VE_1[5], fmt='.', linewidth=1, linestyle='', color='black', label='VE')
plt.errorbar(a, Colle_1[4], yerr=Colle_1[5], fmt='.', linewidth=1, linestyle='', color='green', label='Colle')
plt.errorbar(a, Alpen_1[4], yerr=Alpen_1[5], fmt='.', linewidth=1, linestyle='', color='turquoise', label='Alpen')
plt.errorbar(a, Sammelprobe_1[4], yerr=Sammelprobe_1[5], fmt='.', linewidth=1, linestyle='', color='blue', label='Sammelprobe')
plt.errorbar(b, VE_2[4], yerr=VE_2[5], fmt='.', linewidth=1, linestyle='', color='black')
plt.errorbar(b, Colle_2[4], yerr=Colle_2[5], fmt='.', linewidth=1, linestyle='', color='green')
plt.errorbar(b, Alpen_2[4], yerr=Alpen_2[5], fmt='.', linewidth=1, linestyle='', color='turquoise')
plt.errorbar(b, Sammelprobe_2[4], yerr=Sammelprobe_2[5], fmt='.', linewidth=1, linestyle='', color='blue')
plt.xlabel('Nummer', fontsize=13)
plt.ylabel('Delta_17O', fontsize=13)
plt.title('Fig. [1]:Delta_17O Standards', fontsize=16)
plt.legend()
#plt.show()
plt.close()

print(delta_D_40_1)
print(fraction_40)
plt.errorbar(fraction_40,delta_D_40_1,yerr=delta_D_err_40_1, fmt='.',color='red')
plt.errorbar(fraction_40,delta_D_40_2,yerr=delta_D_err_40_2, fmt='.',color='black')
plt.xlabel('Fraction')
plt.ylabel('Delta_D 40')
plt.title('Delta_D bei 40 Grad')
plt.show()
plt.close()

########################
######Teil 2############
########################

Time, H2O_ppm, H2O_ppm_sd, O18_del, O18_del_sd, D_del, D_del_sd, O17_del, O17_del_sd= np.genfromtxt('data/teil2.txt', delimiter=",", skip_header=1265, skip_footer=700, usecols=(0,1,2,3,4,5,6,7,8), unpack=True)

print(len(H2O_ppm))
time=np.arange(0,len(H2O_ppm))*10.035/60

plt.errorbar(time, H2O_ppm, yerr=H2O_ppm_sd, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeit [min]', fontsize=13)
plt.ylabel('H2O [ppm]', fontsize=13)
plt.title('Fig. [1]:H2O [ppm]', fontsize=16)
#plt.show()
plt.close()

plt.errorbar(time, O18_del, yerr=O18_del_sd, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeit [min]', fontsize=13)
plt.ylabel('O18_del ', fontsize=13)
plt.title('Fig. [1]:', fontsize=16)
#plt.show()
plt.close()

plt.errorbar(time, D_del, yerr=D_del_sd, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeit [min]', fontsize=13)
plt.ylabel('D_del', fontsize=13)
plt.title('Fig. [1]:', fontsize=16)
#plt.show()
plt.close()

plt.errorbar(time, O17_del, yerr=O17_del_sd, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeit [min]', fontsize=13)
plt.ylabel('O17_del', fontsize=13)
plt.title('Fig. [1]:', fontsize=16)
#plt.show()
plt.close()