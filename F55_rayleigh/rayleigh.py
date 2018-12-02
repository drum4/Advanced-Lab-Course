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









##############
#Teil 1#
##############
Delta_2H, Delta_2H_StDev, Delta_18O, Delta_18O_StDev, Delta_17O, Delta_17O_StDev= np.loadtxt('data/teil1_40_50-Processed.txt', usecols=(2,3,4,5,6,7), skiprows=1, unpack=True)

def auslesen(a,b):
    x=np.array([[],[],[],[],[],[]])
    for i in [a,b]:
         x=np.insert(x,0,[[Delta_2H[i], Delta_2H_StDev[i], Delta_18O[i], Delta_18O_StDev[i], Delta_17O[i], Delta_17O_StDev[i]]],axis=1)
    return x

Degre40_1252=auslesen(6,38)#tatsaeliche Zeilennummer aus txt Datei -2
Degre40_1323=auslesen(8,40)
Degre40_1352=auslesen(10,42)
Degre40_1423=auslesen(18,50)
Degre40_1453=auslesen(20,52)
Degre40_1521=auslesen(28,60)
Degre40_1552=auslesen(30,62)
Degre50_1252=auslesen(7,39)
Degre50_1324=auslesen(9,41)
Degre50_1353=auslesen(17,49)
Degre50_1424=auslesen(19,51)
Degre50_1455=auslesen(21,53)
Degre50_1522=auslesen(29,61)
Degre50_1553=auslesen(31,63)

def auslesenstandard(a,b,c,d,e,f,g):
    x=np.array([[],[],[],[],[],[]])
    for i in [a,b,c,d,e,f,g]:
         x=np.insert(x,0,[[Delta_2H[i], Delta_2H_StDev[i], Delta_18O[i], Delta_18O_StDev[i], Delta_17O[i], Delta_17O_StDev[i]]],axis=1)
    return x

VE=auslesenstandard(0,15,27,32,47,59,68)
print('VE=',VE)
print(Degre40_1252)


a=np.arange(1,86)
plt.errorbar(a, Delta_2H,yerr=Delta_2H_StDev, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Nummer', fontsize=13)
plt.ylabel('Delta_2H', fontsize=13)
plt.title('Fig. [1]:Delta_2H', fontsize=16)
plt.show()
plt.close()