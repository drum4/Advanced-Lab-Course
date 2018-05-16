'''
Created on 06.05.2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#import scipy.optimize as optimization
#from scipy.stats import chi2
from matplotlib import rc
#import matplotlib.mlab as mlab
import math

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2




#################
#Photopeak bei verschiedenen Spannungen#
#################
V=np.array([450, 435, 420, 405, 380, 370])
dV=np.array([1, 1, 1, 1, 1, 1])
P1=(np.array([870, 737, 556, 410, 242, 193]))
P=P1**(1/2)
dP1=np.array([10, 20, 20, 15, 10, 10])
dP=1/(2*P1**(1/2))*dP1
M=np.arange(350,500,1)

def linear (x,a,b):
    return a*x+b

popt, pcov = curve_fit(linear, V, P, absolute_sigma=True)
chisquare = np.sum(((linear(V,*popt)-P)**2/dP**2))
dof = 4
chisquare_red = chisquare/dof
plt.errorbar(V, P, xerr=dV, yerr=dP,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.xlabel('Spannung [V]', fontsize=13)
plt.ylabel('Wurzel der Pulshöhe [$\\sqrt{Channel}$]', fontsize=13)
plt.title('Abb. [16]: Position des Photopeaks', fontsize=16)
plt.plot(M, linear(M,*popt), color='red', label='Linearer Fit')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_16.pdf',format='pdf')
#plt.show()
plt.close()





####################
#Energiekalibrierung#
####################

m_e=511.
alpha=1./137
Z=39
pi=math.pi

def Fermifunction (x):
    "Fermifunktion/Coulombkorrektur"
    if(x==0):
        return 0
    F=2*np.pi*(Z*(x+m_e)/(1/alpha*math.sqrt(x*x+2*m_e*x)))/(1-math.exp(-2*np.pi*(38*(x+m_e)/(1/alpha*math.sqrt(x*x+2*m_e*x)))))
    return F


def f80_readfile(filename):
    "read file contents and return numpy array with file contents or, in case of corrected beta spectrum, with Kurie-transformed file contents."
    data=[]
    y_uncertainty=[]
    for line in open(filename): #open file and read file contents
        if(len(line.split())>100):
            temp = line.replace(",", ".")
            columns = [float(x) for x in temp.split()]
            #columns = temp.split(    ) # "    " is the TAB symbol
            dataarray=np.array(columns)
            break
        else:
            columns = line.split(    ) # "    " is the TAB symbol
            energy=float(columns[0])
            counts=float(columns[2])
            if(counts==0):
                data.append(0)
                y_uncertainty.append(0)
            else:
                # conversion to fermi variable
                KuriePlotData=math.sqrt(counts/(Fermifunction(energy)*(energy+m_e)*math.sqrt(energy*energy+2*energy*m_e)))
                sigmaKuriePlotData = math.sqrt(1./(counts*Fermifunction(energy)*(energy+m_e)*math.sqrt(energy*energy+2*energy*m_e))) * math.sqrt(counts) # gaussian error propagation, ussing possonian error on counts
                data.append(KuriePlotData)
                y_uncertainty.append(sigmaKuriePlotData)
            dataarray=np.array(data)
    return dataarray #convert data list to numpy array


SR90 = f80_readfile('data//Sr90_diff.dat')
background = f80_readfile('data//Sr90_t5min_plexi_raw.dat')


#Energiekalibration

xdata = np.array([0.66166,1.3325]) #real energy values
ydata = np.array([250,495]) # measured channels
y_uncertainty = np.array([5,10])

def func(x, a):
    return a*x

fit_parameter, covariance_matrix = curve_fit(func, xdata, ydata,absolute_sigma=True, sigma=y_uncertainty)
steigung = "$m=$ ("+str(np.round(fit_parameter[0],1))+'$\\pm$ '+str(np.round(np.sqrt(covariance_matrix[0][0]),1))+') 1/MeV'
pointplot = plt.errorbar(xdata, ydata, y_uncertainty,
                         fmt='.', linewidth=1,
                         linestyle='', color='black',
                         label='Messpunkte mit Fehler')
x = np.linspace(0, max(xdata))
fitplot  = plt.plot(x, func(x, fit_parameter[0]), 'r-',
                    color='red', label='Linearer Fit')
plt.text(0.7,150,'%s'%(steigung),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.xlabel('Energie [MeV]', fontsize=13)
plt.ylabel('Pulsposition [Channel]', fontsize=13)
plt.title('Abb. [17]: Energiekalibrierung', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_17.pdf',format='pdf')
#plt.show()
plt.close()







##############
#Kurie-Plot#
##############
m_e=511.
alpha=1./137
Z=39
pi=math.pi

####
####
#Sr90#
########
ydataarray=f80_readfile('data//Sr90_diff.dat')
a=0.00267 #calibration slope. Default value (no calibration): a=1
b=0 
xdataarray = np.arange(0,len(ydataarray))*a+b

def linfunc(x, a, b):
    return a*x + b

fitxmin=300
fitxmax=590

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)

x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-',
                    color='red', label='Linearer Fit')

plt.plot(xdataarray,ydataarray,'r-',
         color='black', label='Messkurve')
plt.ylabel('Kurie Variable', fontsize=13)
plt.xlabel('Energie [MeV]', fontsize=13)
xmin=0
xmax=2
plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray),
         'Endpoint Energy = '+str(round(endpointEnergy,2))+' MeV',
         fontsize=16)
plt.title('Abb. [18]: Kurieplot von $^{90}Sr$-Spektrum',
          fontsize=16)
E1=endpointEnergy
plt.legend(frameon=True, loc='right', fontsize = 12)
plt.savefig('figures//f80_abb_18.pdf',format='pdf')
#plt.show()
plt.close()


#########
#Sr90_al5#
#########

ydataarray=f80_readfile('data//Sr90_t5min_al5_raw.dat')-background
b=0
xdataarray = np.arange(0,len(ydataarray))*a+b

#fitrange
fitxmin=250
fitxmax=450

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)

x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-',
                    color='red', label='Linearer Fit')

plt.plot(xdataarray,ydataarray,'r-',
         color='black', label='Messkurve')
plt.ylabel('Kurie Variable', fontsize=13)
plt.xlabel('Energy', fontsize=13)
plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray),
         'Endpoint Energy = '+str(round(endpointEnergy,2))+' MeV',
         fontsize=16)
plt.title('Abb. [19]: Kurieplot von $^{90}Sr$ Spektrum mit 0.5mm Al',
          fontsize=16)
E2=endpointEnergy
plt.legend(frameon=True, loc='right', fontsize = 12)
plt.savefig('figures//f80_abb_19.pdf',format='pdf')
#plt.show()
plt.close()


#Sr90_al10
ydataarray=f80_readfile('data//Sr90_t5min_al10_raw.dat')-background
b=0
xdataarray = np.arange(0,len(ydataarray))*a+b

#fitrange
fitxmin=200
fitxmax=390

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)

x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-',
                    color='red', label='Linearer Fit')

plt.plot(xdataarray,ydataarray,'r-',
         color='black', label='Messkurve')
plt.ylabel('Kurie Variable', fontsize=13)
plt.xlabel('Energy', fontsize=13)

plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray),
         'Endpoint Energy = '+str(round(endpointEnergy,2))+' MeV',
         fontsize=16)
plt.title('Abb. [20]: Kurieplot von $^{90}Sr$ Spektrum mit 1mm Al',
          fontsize=16)
E3=endpointEnergy
plt.legend(frameon=True, loc='right', fontsize = 12)
plt.savefig('figures//f80_abb_20.pdf',format='pdf')
#plt.show()
plt.close()

#Sr90_Al15'
ydataarray=f80_readfile('data//Sr90_t5min_al15_raw.dat')-background
b=0
xdataarray = np.arange(0,len(ydataarray))*a+b

#fitrange
fitxmin=105
fitxmax=350

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)

x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-',
                    color='red', label='Linearer Fit')

plt.plot(xdataarray,ydataarray,'r-',
         color='black', label='Messkurve')
plt.ylabel('Kurie Variable', fontsize=13)
plt.xlabel('Energy', fontsize=13)

plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray), 'Endpoint Energy = '+str(round(endpointEnergy,2))+' MeV', fontsize=16)
plt.title('Abb. [21]: Kurieplot von $^{90}Sr$ Spektrum mit 1.5mm Al', fontsize=16)
E4=endpointEnergy
plt.legend(frameon=True, loc='right', fontsize = 12)
plt.savefig('figures//f80_abb_21.pdf',format='pdf')
#plt.show()
plt.close()


##Sr90_Al30
ydataarray=f80_readfile('data//Sr90_t5min_al30_raw.dat')-background
b=0 #b wird immer wieder neu definiert
xdataarray = np.arange(0,len(ydataarray))*a+b

#fitrange
fitxmin=105
fitxmax=300

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)

x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-',
                    color='red', label='Linearer Fit')

plt.plot(xdataarray,ydataarray,'r-',
         color='black', label='Messkurve')
plt.ylabel('Kurie Variable', fontsize=13)
plt.xlabel('Energy', fontsize=13)

plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray),
         'Endpoint Energy = '+str(round(endpointEnergy,2))+' MeV',
         fontsize=16)
plt.title('Abb. [22]: Kurieplot von $^{90}Sr$ Spektrum mit 3mm Al',
          fontsize=16)
plt.legend(frameon=True, loc='right', fontsize = 12)
E5=endpointEnergy
plt.savefig('figures//f80_abb_22.pdf',format='pdf')
#plt.show()
plt.close()






##############
#Maximalenergie#
###############

d=np.array([1, 1.5, 2, 2.5, 4])
E=np.array([E1, E2, E3, E4, E5])
E_err=E*0.05
popt, pcov = curve_fit(linear, d[:-1], E[:-1], absolute_sigma=True, sigma=E_err[:-1])
chisquare = np.sum(((linear(d,*popt)-E)**2/E_err**2))
dof = 2
chisquare_red = chisquare/dof
abschnitt = '$y_0=($'+str(np.round(popt[1],2))+' $\\pm$ '+str(np.round(np.sqrt(pcov[1][1]),2))+'$)$ MeV'
chisquare_text = '$\\chi_{red}^2=$'+str(np.round(chisquare_red,1))
plt.errorbar(d, E, xerr=E_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.xlabel('Spannung [V]', fontsize=13)
plt.ylabel('Pulshöhe [Channel]', fontsize=13)
plt.title('Abb. [23]: Maximalenergie bei 0mm Absorberdicke', fontsize=16)
plt.plot(d, linear(d,*popt), color='red', label='Linearer Fit')
plt.text(2.74,1.2,'%s \n%s'%(abschnitt,chisquare_text),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_23.pdf',format='pdf')
#plt.show()
plt.close()





###############
#Zeitkalibrierung#
###############

t=np.array([16,18,20,24,26,30])
ch = np.array([276,314,354,435,474,519])
ch_err = np.ones(6)

popt, pcov = curve_fit(linear, t[:-1], ch[:-1],
                       absolute_sigma=True,
                       sigma=ch_err[:-1])
chisquare = np.sum(((linear(t,*popt)-ch)**2/ch_err**2))
dof = 3
chisquare_red = chisquare/dof
steigung = '$m=($'+str(np.round(popt[0],2))+' $\\pm$ '+str(np.round(np.sqrt(pcov[0][0]),2))+'$)$ 1/ns'
y0 = '$y_0=($'+str(np.round(popt[1],1))+' $\\pm$ '+str(np.round(np.sqrt(pcov[1][1]),1))+'$)$'
chisquare_text = '$\\chi_{red}^2=$'+str(np.round(chisquare_red,1))
plt.errorbar(t, ch, yerr=ch_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.xlabel('Delay [ns]', fontsize=13)
plt.ylabel('Kanal', fontsize=13)
plt.title('Abb. [24]: Zeitkalibrierung', fontsize=16)
plt.plot(t, linear(t,*popt), color='red', label='Linearer Fit')
plt.text(24,300,'%s \n%s \n%s'%(steigung,y0,chisquare_text),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_24.pdf',format='pdf')
#plt.show()
plt.close()



##############
#Cosinus-Fit#
##############
raw = np.loadtxt('data//degree0_t901_raw.dat',
                 delimiter = '\t', unpack=True)
a=0
for i in range(100,451):
    a = a + raw[i]

raw = np.loadtxt('data//degree22_t900_raw.dat',
                 delimiter = '\t', unpack=True)
b=0
for i in range(100,451):
    b = b + raw[i]

raw = np.loadtxt('data//degree45_t900_raw.dat',
                 delimiter = '\t', unpack=True)
c=0
for i in range(100,451):
    c = c + raw[i]

raw = np.loadtxt('data//degree67_t900_raw.dat',
                 delimiter = '\t', unpack=True)
d=0
for i in range(100,451):
    d = d + raw[i]

raw = np.loadtxt('data//degree90_t902_raw.dat',
                 delimiter = '\t', unpack=True)
e=0
for i in range(100,451):
    e = e + raw[i]

counts_raw = np.array([a,b,c,d,e])
counts_raw_err =np.sqrt(counts_raw)
counts = counts_raw-900*0.00417
counts_err = np.sqrt(counts_raw_err**2+(900*0.00005)**2)
theta = np.array([0,22,45,67,90])

def cosinus (x,a,b):
    return a*np.cos(b*x/360*2*np.pi)**2

popt, pcov = curve_fit(cosinus, theta, counts,
                       absolute_sigma=True,
                       sigma=counts_err, p0=[2500,1])
chisquare = np.sum(((cosinus(theta,*popt)-counts)**2/counts_err**2))
dof = 3
chisquare_red = chisquare/dof
chisquare_text = '$\\chi_{red}^2=$'+str(np.round(chisquare_red,1))
ab = '$A=$'+str(np.round(popt[0],1))+' $\\pm$ '+str(np.round(np.sqrt(pcov[0][0]),1))
ba = '$B=$'+str(np.round(popt[1],4))+' $\\pm$ '+str(np.round(np.sqrt(pcov[1][1]),4))
plt.errorbar(theta, counts, yerr=counts_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.xlabel('Winkel $\\theta$ [°]', fontsize=13)
plt.ylabel('Zählrate', fontsize=13)
plt.title('Abb. [25]: Winkelabhängigkeit der Myonendetektion', fontsize=16)
M = np.arange(0,91,0.1)
plt.plot(M, cosinus(M,*popt), color='red', label='Linearer Fit')
plt.text(60,2000,'%s \n%s \n%s'%(ab,ba,chisquare_text),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_25.pdf',format='pdf')
#plt.show()
plt.close()