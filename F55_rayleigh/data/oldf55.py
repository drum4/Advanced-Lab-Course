# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:40:54 2018

@author: FP Rayleigh
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit


data1 = np.genfromtxt('data1.txt', skip_header = 1)

def func(frac, a, e):
    return a*frac**e -1

Delta2H_40 = data1[0:7,2]
Delta18O_40 = data1[0:7,4]
Delta17O_40 = data1[0:7,6]

Delta2H_40_2 = data1[7:14,2]
Delta18O_40_2 = data1[7:14,4]
Delta17O_40_2 = data1[7:14,6]

Delta2H_40_ges = np.append(Delta2H_40, Delta2H_40_2)
Delta18O_40_ges = np.append(Delta18O_40, Delta18O_40_2)
Delta17O_40_ges = np.append(Delta17O_40, Delta17O_40_2)

Delta2H_40_mittel = (Delta2H_40 + Delta2H_40_2)/2
Delta18O_40_mittel = (Delta18O_40 + Delta18O_40_2)/2
Delta17O_40_mittel = (Delta17O_40 + Delta17O_40_2)/2

Delta2H_60 = data1[14:21,2]
Delta18O_60 = data1[14:21,4]
Delta17O_60 = data1[14:21,6]

Delta2H_60_2 = data1[21:28,2]
Delta18O_60_2 = data1[21:28,4]
Delta17O_60_2 = data1[21:28,6]

Delta2H_60_ges = np.append(Delta2H_60, Delta2H_60_2)
Delta18O_60_ges = np.append(Delta18O_60, Delta18O_60_2)
Delta17O_60_ges = np.append(Delta17O_60, Delta17O_60_2)

Delta2H_60_mittel = (Delta2H_60 + Delta2H_60_2)/2
Delta18O_60_mittel = (Delta18O_60 + Delta18O_60_2)/2
Delta17O_60_mittel = (Delta17O_60 + Delta17O_60_2)/2


Delta2H_60_mittel[1] = Delta2H_60_2[1]
Delta18O_60_mittel[1] = Delta18O_60_2[1]
Delta17O_60_mittel[1] = Delta17O_60_2[1]



gewicht_leer_1 = 432.2
gewicht_leer_2 = 426.5

gewicht_40_ges = np.array([1058.7, 1049.1, 1038.4, 1027.7, 1017.7, 1007.1, 994.9])
gewicht_60_ges = np.array([1024.2, 994.1, 959.6, 925.2, 892.2, 856.8, 820.3])

gewicht_40 = gewicht_40_ges - gewicht_leer_1
gewicht_60 = gewicht_60_ges - gewicht_leer_2

wasser_40 = 1065.9 - 432.2
wasser_60 = 1042.8 - 426.5

frac_of_water_40 = gewicht_40 / wasser_40
frac_of_water_60 = gewicht_60 / wasser_60

frac_of_water_40_ges = np.append(frac_of_water_40, frac_of_water_40)
frac_of_water_60_ges = np.append(frac_of_water_60, frac_of_water_60)

print('40 Delta_2H')
plt.plot(frac_of_water_40, Delta2H_40_mittel/1000, 'bo')
popt, pcov =curve_fit(func, frac_of_water_40, Delta2H_40_mittel/1000)
print(popt)
y = func(frac_of_water_40, *popt)
plt.gca().invert_xaxis()
plt.plot(frac_of_water_40, y)
plt.show()

print('40 Delta_18O')
plt.plot(frac_of_water_40, Delta18O_40_mittel/1000, 'bo')
popt, pcov =curve_fit(func, frac_of_water_40, Delta18O_40_mittel/1000)
print(popt)
y = func(frac_of_water_40, *popt)
plt.plot(frac_of_water_40, y)
plt.gca().invert_xaxis()
plt.show()

print('40 Delta_17O')
plt.plot(frac_of_water_40, Delta17O_40_mittel/1000, 'bo')
popt, pcov =curve_fit(func, frac_of_water_40, Delta17O_40_mittel/1000)
print(popt)
y = func(frac_of_water_40, *popt)
plt.plot(frac_of_water_40, y)
plt.gca().invert_xaxis()
plt.show()


plt.plot(frac_of_water_60, Delta2H_60_mittel/1000, 'bo')
popt, pcov =curve_fit(func, frac_of_water_60, Delta2H_60_mittel/1000)
print(popt)
y = func(frac_of_water_60, *popt)
plt.plot(frac_of_water_60, y)
plt.gca().invert_xaxis()
plt.show()

plt.plot(frac_of_water_60, Delta18O_60_mittel/1000, 'bo')
popt, pcov =curve_fit(func, frac_of_water_60, Delta18O_60_mittel/1000)
print(popt)
y = func(frac_of_water_60, *popt)
plt.plot(frac_of_water_60, y)
plt.gca().invert_xaxis()
plt.show()

plt.plot(frac_of_water_60, Delta17O_60_mittel/1000, 'bo')
popt, pcov =curve_fit(func, frac_of_water_60, Delta17O_60_mittel/1000)
print(popt)
y = func(frac_of_water_60, *popt)
plt.plot(frac_of_water_60, y)
plt.gca().invert_xaxis()
plt.show()



#print('40')
#plt.plot(frac_of_water_40, Delta2H_40, 'bo')
#plt.plot(frac_of_water_40, Delta2H_40_2, 'ro')
#popt, pcov =curve_fit(func, frac_of_water_40, Delta2H_40)
#print(popt)
#y = func(frac_of_water_40, *popt)
#plt.plot(frac_of_water_40, y)
#plt.show()
#
#plt.plot(frac_of_water_40, Delta18O_40, 'bo')
#plt.plot(frac_of_water_40, Delta18O_40_2, 'ro')
#popt, pcov =curve_fit(func, frac_of_water_40, Delta18O_40)
#y = func(frac_of_water_40, *popt)
#plt.plot(frac_of_water_40, y)
#plt.show()
#
#plt.plot(frac_of_water_40, Delta17O_40, 'bo')
#plt.plot(frac_of_water_40, Delta17O_40_2, 'ro')
#plt.show()
#
#print('60')
#plt.plot(frac_of_water_60, Delta2H_60, 'bo')
#plt.plot(frac_of_water_60, Delta2H_60_2, 'ro')
#plt.show()
#plt.plot(frac_of_water_60, Delta18O_60, 'bo')
#plt.plot(frac_of_water_60, Delta18O_60_2, 'ro')
#plt.show()
#plt.plot(frac_of_water_60, Delta17O_60, 'bo')
#plt.plot(frac_of_water_60, Delta17O_60_2, 'ro')
#plt.show()