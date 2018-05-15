'''
Created on 06.05.2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.optimize as optimization
from scipy.stats import chi2
from matplotlib import rc
import matplotlib.mlab as mlab
import math

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2


#################
#Photopeak bei verscheidenen Spannungen#
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
print('$\\chi_{red}^2$ = '+str(np.round(chisquare_red,1)))
plt.errorbar(V, P, xerr=dV, yerr=dP,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.xlabel('Spannung [V]', fontsize=13)
plt.ylabel('Pulshohe [Channel]', fontsize=13)
plt.title('Abb. [16]: Position des Photopeaks', fontsize=16)
plt.plot(M, linear(M,*popt), color='red', label='Linearer Fit')
plt.savefig('figures//f80_abb_16.pdf',format='pdf')
#plt.show()
plt.close()


################
#Energiekalibrierung#
################

ch = np.array([250,495])
ch_err = np.array([5,10])
E = np.array([0.66166,1.3325])
E_err = np.array([0.00003,0.0003])
M = np.arange(1025)

def prop (x,a):
    return a*x

popt, pcov = curve_fit(prop, ch, E,
                        absolute_sigma=True)
chisquare = np.sum(((prop(ch,*popt)-E)**2/E_err**2))
dof = 1
chisquare_red = chisquare/dof
prob=np.round(1-chi2.cdf(chisquare,dof),2)*100
print("Wahrscheinlichkeit Plot=", prob,"%")

steigung = 'm ='+str(np.round(popt[0],4))+' $\\pm$ '+str(np.round(np.sqrt(pcov[0][0]),4))+' MeV'
chisquare_text = '$\\chi_{red}^2$ = '+str(np.round(chisquare_red,1))

plt.errorbar(ch, E, xerr=ch_err, fmt=".", linewidth=1,
              linestyle='', color='black',
               label='Messpunkte mit Fehler')
plt.xlabel('Pulshöhe [Channel]', fontsize=13)
plt.ylabel('Energie [MeV]', fontsize=13)
plt.title('Abb. [17]: Energie als Funktion der Kanäle',
           fontsize=16)
plt.plot(M, prop(M,*popt),
         color='red', label='Linearer Fit')
plt.text(500,0.5,'%s \n%s'%(steigung,chisquare_text),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_17.pdf',format='pdf')
#plt.show()
plt.close()

##############
#Endpunktenergie beta-Strahlung#
##############


#some constants:
# electron mass in keV
m_e=511.
# Fine-structure constant
alpha=1./137
# Atomic number Strontium
Z=39
# pi
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
    #y_uncertaintyarray=np.array(y_uncertainty)

    
# cal    
def chisquareValue(xvalue, yvalues, sigma_y, fit_parameter, function):
    "return chi2/ndf value of fit for parameter function 'function'"
    chi2=0
    for i in range(len(xvalue)):
        expected = function(xvalue[i],*fit_parameter)
        chi2+=((expected-yvalues[i])/sigma_y[i])**2
    return chi2/(len(xvalue)-len(fit_parameter)) #chis2 per degree of freedom


SR90 = f80_readfile('data//Sr90_diff.dat')
background = f80_readfile('data//Sr90_t5min_plexi_raw.dat')


#Energiekalibration

xdata = np.array([0.66166,1.3325]) #real energy values of Co60_1, Co60_2, Cs137, Mn54, Na22
ydata = np.array([250,495]) # measured channels
y_uncertainty = np.array([5,10])

# Initial guess.


def func(x, a):
    return a*x



fit_parameter, covariance_matrix =curve_fit(func, xdata, ydata,absolute_sigma=True, sigma=y_uncertainty)
steigung = "$m=$ ("+str(np.round(fit_parameter[0],1))+'$\\pm$ '+str(np.round(np.sqrt(covariance_matrix[0][0]),1))+') 1/MeV'
pointplot = plt.errorbar(xdata, ydata, y_uncertainty, linestyle="None")
x = np.linspace(0, max(xdata))
fitplot  = plt.plot(x, func(x, fit_parameter[0]), 'r-', label='fit')
plt.text(0.8,150,'%s'%(steigung),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
#plt.show()

print("E="+str(np.round(fit_parameter[0],1))+' $\\pm$ '+str(np.round(np.sqrt(covariance_matrix[0][0]),1)))
print("chi2/ndf = ",chisquareValue(xdata,ydata,y_uncertainty,fit_parameter,func))


#some constants:
# electron mass in keV
m_e=511.
# Fine-structure constant
alpha=1./137
# Atomic number Strontium
Z=39
# pi
pi=math.pi

ydataarray=f80_readfile('data//Sr90_diff.dat')
#y_uncertaintyarray=np.sqrt(ydataarray)

#Your energy calibration of form y= a*x +b
a=0.00267 #calibration slope. Default value (no calibration): a=1
b=0 #calibration offset. Default value b=0
xdataarray = np.arange(0,len(ydataarray))*a+b


##### fit
# Initial guess.


def linfunc(x, a, b):
    return a*x + b


#fitrange
fitxmin=300
fitxmax=580

#fit_parameter, covariance_matrix = optimization.curve_fit(linfunc, xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax], x0,y_uncertaintyarray[fitxmin:fitxmax])

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)


#pointplot = plt.errorbar(xdataarray, ydataarray, y_uncertaintyarray, linestyle="None")
x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-', label='fit')
####### endfit

plt.plot(xdataarray,ydataarray)
#plt.ylabel('Kurie Variable')
#plt.xlabel('Energy')
xmin=0
xmax=2.2
plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray), 'Endpoint Enegy = '+str(round(endpointEnergy,2))+' Unit', fontsize=16)

plt.show()


print("Endpoint Energy = ", endpointEnergy)

plt.close()


ydataarray=f80_readfile('data//Sr90_t5min_al5_raw.dat')-background
#y_uncertaintyarray=np.sqrt(ydataarray)

#Your energy calibration of form y= a*x +b
a=0.00267 #calibration slope. Default value (no calibration): a=1
b=0 #calibration offset. Default value b=0
xdataarray = np.arange(0,len(ydataarray))*a+b

#fitrange
fitxmin=290
fitxmax=490

#fit_parameter, covariance_matrix = optimization.curve_fit(linfunc, xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax], x0,y_uncertaintyarray[fitxmin:fitxmax])

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)


#pointplot = plt.errorbar(xdataarray, ydataarray, y_uncertaintyarray, linestyle="None")
x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-', label='fit')
####### endfit

plt.plot(xdataarray,ydataarray)
#plt.ylabel('Kurie Variable')
#plt.xlabel('Energy')
xmin=0
xmax=2.2
plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray), 'Endpoint Enegy = '+str(round(endpointEnergy,2))+' Unit', fontsize=16)

plt.show()


print("Endpoint Energy = ", endpointEnergy)

plt.close()



ydataarray=f80_readfile('data//Sr90_t5min_al10_raw.dat')-background
#y_uncertaintyarray=np.sqrt(ydataarray)
a=0.00267
b=0
#Your energy calibration of form y= a*x +b
#calibration offset. Default value b=0
xdataarray = np.arange(0,len(ydataarray))*a+b

#fitrange
fitxmin=200
fitxmax=430

#fit_parameter, covariance_matrix = optimization.curve_fit(linfunc, xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax], x0,y_uncertaintyarray[fitxmin:fitxmax])

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)


#pointplot = plt.errorbar(xdataarray, ydataarray, y_uncertaintyarray, linestyle="None")
x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-', label='fit')
####### endfit

plt.plot(xdataarray,ydataarray)
#plt.ylabel('Kurie Variable')
#plt.xlabel('Energy')

plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray), 'Endpoint Enegy = '+str(round(endpointEnergy,2))+' Unit', fontsize=16)

plt.show()


print("Endpoint Energy = ", endpointEnergy)

plt.close()

ydataarray=f80_readfile('data//Sr90_t5min_al15_raw.dat')-background
#y_uncertaintyarray=np.sqrt(ydataarray)
a=0.00267
b=0
#Your energy calibration of form y= a*x +b
#calibration offset. Default value b=0
xdataarray = np.arange(0,len(ydataarray))*a+b

#fitrange
fitxmin=100
fitxmax=360

#fit_parameter, covariance_matrix = optimization.curve_fit(linfunc, xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax], x0,y_uncertaintyarray[fitxmin:fitxmax])

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)


#pointplot = plt.errorbar(xdataarray, ydataarray, y_uncertaintyarray, linestyle="None")
x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-', label='fit')
####### endfit

plt.plot(xdataarray,ydataarray)
#plt.ylabel('Kurie Variable')
#plt.xlabel('Energy')

plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray), 'Endpoint Enegy = '+str(round(endpointEnergy,2))+' Unit', fontsize=16)

plt.show()


print("Endpoint Energy = ", endpointEnergy)

plt.close()

ydataarray=f80_readfile('data//Sr90_t5min_al30_raw.dat')-background
#y_uncertaintyarray=np.sqrt(ydataarray)
a=0.00267
b=0
#Your energy calibration of form y= a*x +b
#calibration offset. Default value b=0
xdataarray = np.arange(0,len(ydataarray))*a+b

#fitrange
fitxmin=100
fitxmax=300

#fit_parameter, covariance_matrix = optimization.curve_fit(linfunc, xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax], x0,y_uncertaintyarray[fitxmin:fitxmax])

m,b = np.polyfit(xdataarray[fitxmin:fitxmax], ydataarray[fitxmin:fitxmax],1)

print(m,b)


#pointplot = plt.errorbar(xdataarray, ydataarray, y_uncertaintyarray, linestyle="None")
x = np.linspace(0, max(xdataarray), 1000)
fitplot  = plt.plot(x, linfunc(x, m, b), 'r-', label='fit')
####### endfit

plt.plot(xdataarray,ydataarray)
#plt.ylabel('Kurie Variable')
#plt.xlabel('Energy')

plt.xlim((xmin,xmax)) #restrict x axis to [xmin,xmax]
axes = plt.gca()
axes.set_ylim([0,1.1*max(ydataarray)])
endpointEnergy=-b/m
plt.text(xmax*0.4,max(ydataarray), 'Endpoint Enegy = '+str(round(endpointEnergy,2))+' Unit', fontsize=16)

plt.show()


print("Endpoint Energy = ", endpointEnergy)

plt.close()

d=np.array([1, 1.5, 2, 2.5, 4])
E=np.array([1.67871258539, 1.33628874118, 1.16859621833, 0.98276189432, 0.69788234755])
E_err=E*0.05
popt, pcov = curve_fit(linear, d[:-1], E[:-1], absolute_sigma=True, sigma=E_err[:-1])
chisquare = np.sum(((linear(d,*popt)-E)**2/E_err**2))
dof = 2
chisquare_red = chisquare/dof
abschnitt = 'y_0 ='+str(np.round(popt[1],4))+' $\\pm$ '+str(np.round(np.sqrt(pcov[1][1]),4))+' MeV'
chisquare_text = '$\\chi_{red}^2$ = '+str(np.round(chisquare_red,1))
print('$\\chi_{red}^2$ = '+str(np.round(chisquare_red,1)))
plt.errorbar(d, E, xerr=E_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.xlabel('Spannung [V]', fontsize=13)
plt.ylabel('Pulshohe [Channel]', fontsize=13)
plt.title('Abb. [16]: Position des Photopeaks', fontsize=16)
plt.plot(d, linear(d,*popt), color='red', label='Linearer Fit')
plt.text(2.5,1.4,'%s \n%s'%(abschnitt,chisquare_text),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.legend(frameon=True, fontsize = 12)
#plt.savefig('figures//f80_abb_30.pdf',format='pdf')
plt.show()
plt.close()