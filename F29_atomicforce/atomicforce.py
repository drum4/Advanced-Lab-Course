'''
Created on 09.01.2019

@author: timok
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

########################
## Abschnitt 3.1.2 #####
########################

#####TGZ01 old tip#####

print('TGZ01 old tip')
w_2=np.array([5.91, 5.90, 6.08, 5.95, 5.91])
h1=np.array([20.10, 19.43, 18.61, 18.82, 18.49, 20.66, 18.19, 20.26, 19.26, 20.66])
phi1=np.array([8.71, 5.35, 8.71, 6.81, 15.88, 6.23, 14.01, 12.03, 12.12, 4.75])

w=np.mean(w_2)/2
w_err=np.std(w_2)/2
h=np.mean(h1)
h_err=np.std(h1)
phi=np.mean(phi1)
phi_err=np.std(phi)

print('w=',w,'+',w_err)
print('h=',h,'+',h_err)
print('phi=',phi,'+',phi_err)

#####TGZ01 new tip#####

print('TGZ01 new tip')
w_2=np.array([5.87, 5.87, 5.91, 5.92, 5.88])
h1=np.array([19.70, 21.93, 19.74, 21.61, 20.08, 19.11, 20.04, 21.24, 19.19, 20.81])
phi1=np.array([12,46, 12.12, 11.29, 8.23, 9.50, 12.02, 8.44, 13.73, 7.30, 7.56])

w=np.mean(w_2)/2
w_err=np.std(w_2)/2
h=np.mean(h1)
h_err=np.std(h1)
phi=np.mean(phi1)
phi_err=np.std(phi)

print('w=',w,'+',w_err)
print('h=',h,'+',h_err)
print('phi=',phi,'+',phi_err)

##################
#### CCD Chip ####
##################

a=np.array([4.50, 4.57, 4.53])
b=np.array([4.79, 4.78, 4.80])

a_err=np.std(a)
a=np.mean(a)
b_err=np.std(b)
b=np.mean(b)

print('a=',a,'+',a_err)
print('b=',b,'+',b_err)

area=a*b  #[(mu meter)^2]
area_err=np.sqrt((b*a_err)**2+(a*b_err)**2)
print('area=',area,'+',area_err)
area_chip=4000*6000 #[(mu meter)^2]
area_chip_err=np.sqrt((6000*500)**2+(4000*500)**2)
print('area_chip=',area_chip,'+',area_chip_err)

anzahl=area_chip/area
anzahl_err=np.sqrt((area_chip_err/area)**2+(area_chip/area**2*area_err)**2)
print('anzahl=',anzahl,'+',anzahl_err)


###################
### Nano Lattice ##
###################

print('bright')
tiefe=np.array([168.4, 167.9, 168.1])
durchmesser=np.array([0.114, 0.096, 0.092])
abstand=np.array([0.143, 0.167, 0.160])

t=np.mean(tiefe)
t_err=np.std(tiefe)
d=np.mean(durchmesser)
d_err=np.std(durchmesser)
a=np.mean(abstand)
a_err=np.std(abstand)

print('t=',t,'+',t_err)
print('d=',d,'+',d_err)
print('a=',a,'+',a_err)




print('dark')
tiefe=np.array([227.8, 217.1, 225.9])
durchmesser=np.array([0.357, 0.322, 0.356])
abstand=np.array([0.296, 0.237, 0.326])

t=np.mean(tiefe)
t_err=np.std(tiefe)
d=np.mean(durchmesser)
d_err=np.std(durchmesser)
a=np.mean(abstand)
a_err=np.std(abstand)

print('t=',t,'+',t_err)
print('d=',d,'+',d_err)
print('a=',a,'+',a_err)

###############
#### CD #######
###############

pits=np.array([7.65/4, 4.84/2, 1.74, 3.41/2, 4.75/2])
print(pits)
print("2.42/9, 2.375/9, 1.9125/7, 1.74/6, 1.705/6")
print('oder 2.42/7, 2.375/7, 1.74/5, 1.705/5 ')