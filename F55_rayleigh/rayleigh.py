'''
Created on Nov 27, 2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.optimize import curve_fit

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

Degre40_1252=auslesen(6,38)#tatsaeliche Zeilennummer aus txt Datei -2
Degre40_1323=auslesen(8,40)
Degre40_1352=auslesen(10,42)
Degre50_1455=auslesen(21,53)
Degre50_1522=auslesen(29,61)
Degre50_1553=auslesen(31,63)


VE_1=auslesenstandard(0,15,27,32,47,59,68)
Alpen_1=auslesenstandard(2,13,23,34,45,55,66)
Colle_1=auslesenstandard(4,11,26,36,43,58,64)
Sammelprobe_1=auslesenstandard(5,16,25,37,48,57,69)


Delta_2H, Delta_2H_StDev, Delta_18O, Delta_18O_StDev, Delta_17O, Delta_17O_StDev= np.loadtxt('data/teil1_60_teil3_4-Processed.txt', usecols=(2,3,4,5,6,7), skiprows=1, unpack=True)

Degre60_1321=auslesen(20,53)
Degre60_1351=auslesen(21,54)
Degre60_1421=auslesen(28,61)
Degre60_1451=auslesen(29,62)
Degre60_1521=auslesen(30,63)
Degre60_1551=auslesen(31,64)
Degre60_1621=auslesen(32,65)

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
plt.show()
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
plt.show()
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
plt.show()
plt.close()

VE=auslesenstandard(0,15,27,32,47,59,68)
print('VE=',VE)
print(Degre40_1252)


########################
######Teil 2############
########################

#H2O_ppm, H2O_ppm_sd, O18_del, O18_del_sd, D_del, D_del_sd, O17_del, O17_del_sd= np.loadtxt('data/teil2.txt', delimiter=',',  skiprows=3, usecols=(1,2,3,4,5,6,7,8), unpack=True)

#a=np.arange(1,86)
#plt.errorbar(a, Delta_2H,yerr=Delta_2H_StDev, fmt='.', linewidth=1, linestyle='', color='black')
#plt.xlabel('Nummer', fontsize=13)
#plt.ylabel('Delta_2H', fontsize=13)
#plt.title('Fig. [1]:Delta_2H', fontsize=16)
#plt.show()
#plt.close()