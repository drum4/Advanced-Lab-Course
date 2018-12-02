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
VE=np.array([[],[],[],[],[],[]])
Degre40_1252=np.array([[],[],[],[],[],[]])
Degre40_1323=np.array([[],[],[],[],[],[]])
Degre40_1352=np.array([[],[],[],[],[],[]])
Degre40_1423=np.array([[],[],[],[],[],[]])
Degre40_1453=np.array([[],[],[],[],[],[]])
Degre40_1521=np.array([[],[],[],[],[],[]])
Degre40_1552=np.array([[],[],[],[],[],[]])
Degre50_1252=np.array([[],[],[],[],[],[]])
Degre50_1324=np.array([[],[],[],[],[],[]])
Degre50_1353=np.array([[],[],[],[],[],[]])
Degre50_1424=np.array([[],[],[],[],[],[]])
Degre50_1455=np.array([[],[],[],[],[],[]])
Degre50_1522=np.array([[],[],[],[],[],[]])
Degre50_1553=np.array([[],[],[],[],[],[]])


Delta_2H, Delta_2H_StDev, Delta_18O, Delta_18O_StDev, Delta_17O, Delta_17O_StDev= np.loadtxt('data/teil1_40_50-Processed.txt', usecols=(2,3,4,5,6,7), skiprows=1, unpack=True)
def auslesen(x,a,b):
    x=np.array([[],[],[],[],[],[]])
    for i in [a,b]:
         x=np.insert(x,0,[[Delta_2H[a], Delta_2H_StDev[a], Delta_18O[a], Delta_18O_StDev[a], Delta_17O[a], Delta_17O_StDev[a]]],axis=1)
    return x

for i in [0,15,27,32,47,59,68]: #tatsaeliche Zeilennummer aus txt Datei -2
    VE=np.insert(VE,0,[[Delta_2H[i], Delta_2H_StDev[i], Delta_18O[i], Delta_18O_StDev[i], Delta_17O[i], Delta_17O_StDev[i]]],axis=1)

for i in [6,38]:
    Degre40_1252=np.insert(Degre40_1252,0,[[Delta_2H[i], Delta_2H_StDev[i], Delta_18O[i], Delta_18O_StDev[i], Delta_17O[i], Delta_17O_StDev[i]]],axis=1)

for i in [7,39]:
    Degre40_1323=np.insert(Degre40_1323,0,[[Delta_2H[i], Delta_2H_StDev[i], Delta_18O[i], Delta_18O_StDev[i], Delta_17O[i], Delta_17O_StDev[i]]],axis=1)

auslesen(Degre40_1352,10,11)
print('Degre40_1352=',Degre40_1352)

#print(VE)
#print(Degre40_1252)
#print(Degre40_1323)
a=np.arange(1,86)
plt.errorbar(a, Delta_2H,yerr=Delta_2H_StDev, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Nummer', fontsize=13)
plt.ylabel('Delta_2H', fontsize=13)
plt.title('Fig. [1]:Delta_2H', fontsize=16)
plt.show()
plt.close()