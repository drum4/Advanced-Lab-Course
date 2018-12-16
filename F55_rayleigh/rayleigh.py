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
         x=np.insert(x,0,[Delta_2H[i], Delta_2H_StDev[i], Delta_18O[i], Delta_18O_StDev[i], Delta_17O[i], Delta_17O_StDev[i]],axis=1)
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

def longarray(x):
    xarray=np.array([[]])
    for l in range(0,6):
        xarray_1=np.array([])
        for i in range(0,2):
            xarray_2=np.array([])
            for j in range(0,7):
                xarray_2=np.append(xarray_2, [x[j][l][i]])
            xarray_1=np.append(xarray_1, [xarray_2])
        xarray=np.append(xarray, [xarray_1])
    return xarray

def longarraymean(x):
    xarray=np.array([[]])
    for i in range(0,6):
        xarray_1=np.array([])
        for j in range(0,7):
            xarray_1=np.append(xarray_1, [x[j][i]])
        xarray=np.append(xarray, [xarray_1])
    return xarray

def linear(x,a,b):
    return a*x+b

def frac(x,a,b):
    return b*x**(a-1)-1


Degre40_1252=auslesen(6,38)/1000#tatsaeliche Zeilennummer aus txt Datei -2
Degre40_1323=auslesen(8,40)/1000
Degre40_1352=auslesen(10,42)/1000
Degre40_1423=auslesen(18,50)/1000
Degre40_1453=auslesen(20,52)/1000
Degre40_1521=auslesen(28,60)/1000
Degre40_1552=auslesen(30,62)/1000
Degre40=np.array([Degre40_1252,Degre40_1323,Degre40_1352,Degre40_1423,Degre40_1453,Degre40_1521,Degre40_1552])
Degre40=longarray(Degre40)

Degre40_1252_mean=meansample(Degre40_1252)
Degre40_1323_mean=meansample(Degre40_1323)
Degre40_1352_mean=meansample(Degre40_1352)
Degre40_1423_mean=meansample(Degre40_1423)
Degre40_1453_mean=meansample(Degre40_1453)
Degre40_1521_mean=meansample(Degre40_1521)
Degre40_1552_mean=meansample(Degre40_1552)
Degre40_mean=np.array([Degre40_1252_mean,Degre40_1323_mean,Degre40_1352_mean,Degre40_1423_mean,Degre40_1453_mean,Degre40_1521_mean,Degre40_1552_mean])

Degre40_mean=longarraymean(Degre40_mean)

Masse_40=np.array([1038.20,1026.02,1014.70,1003.19,991.83,981.25,969.81])
fraction_40=Masse_40/(Masse_40[0])


Degre50_1252=auslesen(7,39)/1000
Degre50_1324=auslesen(9,41)/1000
Degre50_1353=auslesen(17,49)/1000
Degre50_1424=auslesen(19,51)/1000
Degre50_1455=auslesen(21,53)/1000
Degre50_1522=auslesen(29,61)/1000
Degre50_1553=auslesen(31,63)/1000
Degre50=np.array([Degre50_1252,Degre50_1324,Degre50_1353,Degre50_1424,Degre50_1455,Degre50_1522,Degre50_1553])
Degre50=longarray(Degre50)


Degre50_1252_mean=meansample(Degre50_1252)
Degre50_1324_mean=meansample(Degre50_1324)
Degre50_1353_mean=meansample(Degre50_1353)
Degre50_1424_mean=meansample(Degre50_1424)
Degre50_1455_mean=meansample(Degre50_1455)
Degre50_1522_mean=meansample(Degre50_1522)
Degre50_1553_mean=meansample(Degre50_1553)
Degre50_mean=np.array([Degre50_1252_mean,Degre50_1324_mean,Degre50_1353_mean,Degre50_1424_mean,Degre50_1455_mean,Degre50_1522_mean,Degre50_1553_mean])
Degre50_mean=longarraymean(Degre50_mean)

Masse_50=np.array([1012.61, 990.05, 968.50, 946.43, 925.30, 903.97, 891.91])
fraction_50=Masse_50/(Masse_50[0])


VE_1=auslesenstandard(0,15,27,32,47,59,68)/1000
Alpen_1=auslesenstandard(2,13,23,34,45,55,66)/1000
Colle_1=auslesenstandard(4,11,26,36,43,58,64)/1000
Sammelprobe_1=auslesenstandard(5,16,25,37,48,57,69)/1000


Delta_2H, Delta_2H_StDev, Delta_18O, Delta_18O_StDev, Delta_17O, Delta_17O_StDev= np.loadtxt('data/teil1_60_teil3_4-Processed.txt', usecols=(2,3,4,5,6,7), skiprows=1, unpack=True)

Degre60_1321=auslesen(20,53)/1000
Degre60_1351=auslesen(21,54)/1000
Degre60_1428=auslesen(28,61)/1000
Degre60_1455=auslesen(29,62)/1000
Degre60_1525=auslesen(30,63)/1000
Degre60_1554=auslesen(31,64)/1000
Degre60_1623=auslesen(32,65)/1000
Degre60=np.array([Degre60_1321,Degre60_1351,Degre60_1428,Degre60_1455,Degre60_1525,Degre60_1554,Degre60_1623])
Degre60=longarray(Degre60)

rain293=meansample(auslesen(6,39)/1000)
rain298=meansample(auslesen(8,41)/1000)
rain58=meansample(auslesen(7,40)/1000)


Degre60_1321_mean=meansample(Degre60_1321)
Degre60_1351_mean=meansample(Degre60_1351)
Degre60_1428_mean=meansample(Degre60_1428)
Degre60_1455_mean=meansample(Degre60_1455)
Degre60_1525_mean=meansample(Degre60_1525)
Degre60_1554_mean=meansample(Degre60_1554)
Degre60_1623_mean=meansample(Degre60_1623)
Degre60_mean=np.array([Degre60_1321_mean,Degre60_1351_mean,Degre60_1428_mean,Degre60_1455_mean,Degre60_1525_mean,Degre60_1554_mean,Degre60_1623_mean])
Degre60_mean=longarraymean(Degre60_mean)

Masse_60=np.array([953.10,919.13,875.74,844.71,809.80,776.12,740.07])
fraction_60=Masse_60/(Masse_60[0])


Sample_293=auslesen(6,39)/1000
Sample_58=auslesen(7,40)/1000
Sample_298=auslesen(8,41)/1000
Sample_k009=auslesen(9,42)/1000
Sample_w051=auslesen(10,43)/1000
Sample_w075=auslesen(17,50)/1000
Sample_w023=auslesen(18,51)/1000
Sample_w077=auslesen(19,52)/1000

VE_2=auslesenstandard(0,15,26,33,48,59,70)/1000
Alpen_2=auslesenstandard(2,13,23,35,45,57,68)/1000
Colle_2=auslesenstandard(4,16,27,37,49,55,66)/1000
Sammelprobe_2=auslesenstandard(5,11,25,38,47,60,71)/1000

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

############################


plt.errorbar(fraction_40,(Degre40)[0:7],yerr=(Degre40)[14:21], fmt='.',color='red', label='Messung 2')
plt.errorbar(fraction_40,(Degre40)[7:14],yerr=(Degre40)[21:28], fmt='.',color='black', label='Messung 1')
#plt.errorbar(fraction_40,Degre40_mean[0:7],yerr=Degre40_mean[7:14], fmt='.', color='green', label='Mittelwert')
plt.xlabel('Fraction')
plt.ylabel('Delta_D 40')
plt.title('Delta_D bei 40 Grad')
plt.legend()
plt.gca().invert_xaxis()
#plt.show()
plt.close()

plt.errorbar(fraction_40,(Degre40)[28:35],yerr=(Degre40)[42:49], fmt='.',color='red', label='Messung 2')
plt.errorbar(fraction_40,(Degre40)[35:42],yerr=(Degre40)[49:56], fmt='.',color='black', label='Messung 1')
plt.errorbar(fraction_40,Degre40_mean[14:21],yerr=Degre40_mean[21:28], fmt='.', color='green', label='Mittelwert')
plt.xlabel('Fraction')
plt.ylabel('Delta_O18 40')
plt.title('Delta_O18 bei 40 Grad')
plt.legend()
plt.gca().invert_xaxis()
#plt.show()
plt.close()

plt.errorbar(fraction_40,(Degre40)[56:63],yerr=(Degre40)[70:77], fmt='.',color='red', label='Messung 2')
plt.errorbar(fraction_40,(Degre40)[63:70],yerr=(Degre40)[77:84], fmt='.',color='black', label='Messung 1')
plt.errorbar(fraction_40,Degre40_mean[28:35],yerr=Degre40_mean[35:42], fmt='.', color='green', label='Mittelwert')
plt.xlabel('Fraction')
plt.ylabel('Delta_O17 40')
plt.title('Delta_O17 bei 40 Grad')
plt.legend()
plt.gca().invert_xaxis()
#plt.show()
plt.close()

############

plt.errorbar(fraction_50,(Degre50)[0:7],yerr=(Degre50)[14:21], fmt='.',color='red', label='Messung 2')
plt.errorbar(fraction_50,(Degre50)[7:14],yerr=(Degre50)[21:28], fmt='.',color='black', label='Messung 1')
plt.errorbar(fraction_50,Degre50_mean[0:7],yerr=Degre50_mean[7:14], fmt='.', color='green', label='Mittelwert')
plt.xlabel('Fraction')
plt.ylabel('Delta_D 50')
plt.title('Delta_D bei 50 Grad')
plt.legend()
plt.gca().invert_xaxis()
#plt.show()
plt.close()

plt.errorbar(fraction_50,(Degre50)[28:35],yerr=(Degre50)[42:49], fmt='.',color='red', label='Messung 2')
plt.errorbar(fraction_50,(Degre50)[35:42],yerr=(Degre50)[49:56], fmt='.',color='black', label='Messung 1')
plt.errorbar(fraction_50,Degre50_mean[14:21],yerr=Degre50_mean[21:28], fmt='.', color='green', label='Mittelwert')
plt.xlabel('Fraction')
plt.ylabel('Delta_O18 50')
plt.title('Delta_O18 bei 50 Grad')
plt.legend()
plt.gca().invert_xaxis()
#plt.show()
plt.close()

plt.errorbar(fraction_50,(Degre50)[56:63],yerr=(Degre50)[70:77], fmt='.',color='red', label='Messung 2')
plt.errorbar(fraction_50,(Degre50)[63:70],yerr=(Degre50)[77:84], fmt='.',color='black', label='Messung 1')
plt.errorbar(fraction_50,Degre50_mean[28:35],yerr=Degre50_mean[35:42], fmt='.', color='green', label='Mittelwert')
plt.xlabel('Fraction')
plt.ylabel('Delta_O17 50')
plt.title('Delta_O17 bei 50 Grad')
plt.legend()
plt.gca().invert_xaxis()
#plt.show()
plt.close()

#############

plt.errorbar(fraction_60,(Degre60)[0:7],yerr=(Degre60)[14:21], fmt='.',color='red', label='Messung 2')
plt.errorbar(fraction_60,(Degre60)[7:14],yerr=(Degre60)[21:28], fmt='.',color='black', label='Messung 1')
plt.errorbar(fraction_60,Degre60_mean[0:7],yerr=Degre60_mean[7:14], fmt='.', color='green', label='Mittelwert')
plt.xlabel('Fraction')
plt.ylabel('Delta_D 60')
plt.title('Delta_D bei 60 Grad')
plt.legend()
plt.gca().invert_xaxis()
#plt.show()
plt.close()

plt.errorbar(fraction_60,(Degre60)[28:35],yerr=(Degre60)[42:49], fmt='.',color='red', label='Messung 2')
plt.errorbar(fraction_60,(Degre60)[35:42],yerr=(Degre60)[49:56], fmt='.',color='black', label='Messung 1')
plt.errorbar(fraction_60,Degre60_mean[14:21],yerr=Degre60_mean[21:28], fmt='.', color='green', label='Mittelwert')
plt.xlabel('Fraction')
plt.ylabel('Delta_O18 60')
plt.title('Delta_O18 bei 60 Grad')
plt.legend()
plt.gca().invert_xaxis()
#plt.show()
plt.close()

plt.errorbar(fraction_60,(Degre60)[56:63],yerr=(Degre60)[70:77], fmt='.',color='red', label='Messung 2')
plt.errorbar(fraction_60,(Degre60)[63:70],yerr=(Degre60)[77:84], fmt='.',color='black', label='Messung 1')
plt.errorbar(fraction_60,Degre60_mean[28:35],yerr=Degre60_mean[35:42], fmt='.', color='green', label='Mittelwert')
plt.xlabel('Fraction')
plt.ylabel('Delta_O17 60')
plt.title('Delta_O17 bei 60 Grad')
plt.legend()
plt.gca().invert_xaxis()
#plt.show()
plt.close()

###############

popt1, pcov1 = curve_fit(frac, fraction_40, Degre40_mean[0:7], absolute_sigma=True, sigma=Degre40_mean[7:14])
popt2, pcov2 = curve_fit(frac, fraction_50, Degre50_mean[0:7], absolute_sigma=True, sigma=Degre50_mean[7:14])
popt3, pcov3 = curve_fit(frac, fraction_60, Degre60_mean[0:7], absolute_sigma=True, sigma=Degre60_mean[7:14])
plt.plot(fraction_40, frac(fraction_40,*popt1), linestyle='-', color='blue')
plt.plot(fraction_50, frac(fraction_50,*popt2), linestyle='-', color='green')
plt.plot(fraction_60, frac(fraction_60,*popt3), linestyle='-', color='red')
plt.errorbar(fraction_40,Degre40_mean[0:7],yerr=Degre40_mean[7:14], fmt='.', color='blue', label='40° C')
plt.errorbar(fraction_50,Degre50_mean[0:7],yerr=Degre50_mean[7:14], fmt='.', color='green', label='50° C')
plt.errorbar(fraction_60,Degre60_mean[0:7],yerr=Degre60_mean[7:14], fmt='.', color='red', label='60° C')
plt.xlabel('Fraction', fontsize=13)
plt.ylabel('$\\delta D$', fontsize=13)
plt.title('Abb. [1]: Mittelwert: $\\delta D$',fontsize=16)
plt.gca().invert_xaxis()
plt.legend(frameon=True, fontsize = 12)
plt.tight_layout()
plt.savefig('figures//f55_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

alphaD40=popt1[0]
alphaD40err=pcov1[0][0]
alphaD50=popt2[0]
alphaD50err=pcov2[0][0]
alphaD60=popt3[0]
alphaD60err=pcov3[0][0]
print('alphaD40=',alphaD40,'+',alphaD40err)
print('alphaD50=',alphaD50,'+',alphaD50err)
print('alphaD60=',alphaD60,'+',alphaD60err)

popt1, pcov1 = curve_fit(frac, fraction_40, Degre40_mean[14:21], absolute_sigma=True, sigma=Degre40_mean[21:28])
popt2, pcov2 = curve_fit(frac, fraction_50, Degre50_mean[14:21], absolute_sigma=True, sigma=Degre50_mean[21:28])
popt3, pcov3 = curve_fit(frac, fraction_60, Degre60_mean[14:21], absolute_sigma=True, sigma=Degre60_mean[21:28])
plt.plot(fraction_40, frac(fraction_40,*popt1), linestyle='-', color='blue')
plt.plot(fraction_50, frac(fraction_50,*popt2), linestyle='-', color='green')
plt.plot(fraction_60, frac(fraction_60,*popt3), linestyle='-', color='red')
plt.errorbar(fraction_40,Degre40_mean[14:21],yerr=Degre40_mean[21:28], fmt='.', color='blue', label='40° C')
plt.errorbar(fraction_50,Degre50_mean[14:21],yerr=Degre50_mean[21:28], fmt='.', color='green', label='50° C')
plt.errorbar(fraction_60,Degre60_mean[14:21],yerr=Degre60_mean[21:28], fmt='.', color='red', label='60° C')
plt.xlabel('Fraction', fontsize=13)
plt.ylabel('$\\delta^{18} O$', fontsize=13)
plt.title('Abb. [2]: Mittelwert: $\\delta^{18} O$',fontsize=16)
plt.gca().invert_xaxis()
plt.legend(frameon=True, fontsize = 12)
plt.tight_layout()
plt.savefig('figures//f55_abb_2.pdf',format='pdf')
#plt.show()
plt.close()

alphaO1840=popt1[0]
alphaO1840err=pcov1[0][0]
alphaO1850=popt2[0]
alphaO1850err=pcov2[0][0]
alphaO1860=popt3[0]
alphaO1860err=pcov3[0][0]
print('alphaO1840=',alphaO1840,'+',alphaO1840err)
print('alphaO1850=',alphaO1850,'+',alphaO1850err)
print('alphaO1860=',alphaO1860,'+',alphaO1860err)

popt1, pcov1 = curve_fit(frac, fraction_40, Degre40_mean[28:35], absolute_sigma=True, sigma=Degre40_mean[35:42])
popt2, pcov2 = curve_fit(frac, fraction_50, Degre50_mean[28:35], absolute_sigma=True, sigma=Degre50_mean[35:42])
popt3, pcov3 = curve_fit(frac, fraction_60, Degre60_mean[28:35], absolute_sigma=True, sigma=Degre60_mean[35:42])
plt.plot(fraction_40, frac(fraction_40,*popt1), linestyle='-', color='blue')
plt.plot(fraction_50, frac(fraction_50,*popt2), linestyle='-', color='green')
plt.plot(fraction_60, frac(fraction_60,*popt3), linestyle='-', color='red')
plt.errorbar(fraction_40,Degre40_mean[28:35],yerr=Degre40_mean[35:42], fmt='.', color='blue', label='40° C')
plt.errorbar(fraction_50,Degre50_mean[28:35],yerr=Degre50_mean[35:42], fmt='.', color='green', label='50° C')
plt.errorbar(fraction_60,Degre60_mean[28:35],yerr=Degre60_mean[35:42], fmt='.', color='red', label='60° C')
plt.xlabel('Fraction', fontsize=13)
plt.ylabel('$\\delta^{17} O$', fontsize=13)
plt.title('Abb. [3]: Mittelwert: $\\delta^{17} O$',fontsize=16)
plt.gca().invert_xaxis()
plt.legend(frameon=True, fontsize = 12)
plt.tight_layout()
plt.savefig('figures//f55_abb_3.pdf',format='pdf')
#plt.show()
plt.close()

alphaO1740=popt1[0]
alphaO1740err=pcov1[0][0]
alphaO1750=popt2[0]
alphaO1750err=pcov2[0][0]
alphaO1760=popt3[0]
alphaO1760err=pcov3[0][0]
print('alphaO1740=',alphaO1740,'+',alphaO1740err)
print('alphaO1750=',alphaO1750,'+',alphaO1750err)
print('alphaO1760=',alphaO1760,'+',alphaO1760err)

plt.errorbar(np.array([40,50,60]),np.array([alphaD40,alphaD50,alphaD60]), yerr=np.array([alphaD40err,alphaD50err,alphaD60err]), fmt='.', color='black', label='Deuterium')
plt.errorbar([40,50,60],[alphaO1840,alphaO1850,alphaO1860], yerr=[alphaO1840err,alphaO1850err,alphaO1860err], fmt='.', color='red', label='$^{18}O$')
plt.errorbar([40,50,60],[alphaO1740,alphaO1750,alphaO1760], yerr=[alphaO1740err,alphaO1750err,alphaO1760err], fmt='.', color='blue', label='$^{17}O$')
plt.xlabel('Temperatur [°C]')
plt.ylabel('$\\alpha$')
plt.title('Abb. [5]: $\\alpha$-Werte der Isotope')
plt.legend()
plt.savefig('figures//f55_abb_5.pdf',format='pdf')
#plt.show()
plt.close()

###############
##waterline temperature##
##############


Delta_O18=np.append(Degre40_mean[14:21],[Degre50_mean[14:21],Degre60_mean[14:21]])
Delta_O18_err=np.append(Degre40_mean[21:28],[Degre50_mean[21:28],Degre60_mean[21:28]])
Delta_D=np.append(Degre40_mean[0:7],[Degre50_mean[0:7],Degre60_mean[0:7]])
Delta_D_err=np.append(Degre40_mean[7:14],[Degre50_mean[7:14],Degre60_mean[7:14]])

popt, pcov = curve_fit(linear, Delta_O18, Delta_D, absolute_sigma=True, sigma=Delta_D_err)
popt1, pcov1 = curve_fit(linear, Degre40_mean[14:21], Degre40_mean[0:7], absolute_sigma=True, sigma=Degre40_mean[7:14])
popt2, pcov2 = curve_fit(linear, Degre50_mean[14:21], Degre50_mean[0:7], absolute_sigma=True, sigma=Degre50_mean[7:14])
popt3, pcov3 = curve_fit(linear, Degre60_mean[14:21], Degre60_mean[0:7], absolute_sigma=True, sigma=Degre60_mean[7:14])
#plt.plot(Degre60_mean[14:21], linear(Degre60_mean[14:21],*popt1), linestyle='-', color='blue')
#plt.plot(Degre60_mean[14:21], linear(Degre60_mean[14:21],*popt2), linestyle='-', color='green')
#plt.plot(Degre60_mean[14:21], linear(Degre60_mean[14:21],*popt3), linestyle='-', color='red')
a=popt1[0]
aerr=pcov1[0][0]
b=popt2[0]
berr=pcov2[0][0]
c=popt3[0]
cerr=pcov3[0][0]
d=popt[0]
d_err=pcov[0][0]

print(a)
print(b)
print(c)
slope_einzel=(a+b+c)/3
#slopeerr=np.sqrt(aerr**2+berr**2+cerr**2)/3
slope_einzel_err=np.std([a,b,c])
print('slope_einzel=',slope_einzel,'+',slope_einzel_err)
slope=np.round(d,4)
slope_err=np.round(d_err,4)
print('slope=',slope,'+',slope_err)

plt.plot(Delta_O18, linear(Delta_O18,*popt), linestyle='-', color='black', label='linearer Fit')
plt.errorbar(Degre40_mean[14:21],Degre40_mean[0:7],xerr=Degre40_mean[21:28],yerr=Degre40_mean[7:14], fmt='.', color='blue', label='40° C')
plt.errorbar(Degre50_mean[14:21],Degre50_mean[0:7],xerr=Degre50_mean[21:28],yerr=Degre50_mean[7:14], fmt='.', color='green', label='50° C')
plt.errorbar(Degre60_mean[14:21],Degre60_mean[0:7],xerr=Degre60_mean[21:28],yerr=Degre60_mean[7:14], fmt='.', color='red', label='60° C')
plt.xlabel('$\\delta^{18} O$')
plt.ylabel('$\\delta D$')
plt.title('Abb. [4]: Waterline of different temperatures')
plt.text(-0.003,-0.058,'Steigung: %s $\\pm$ %s'%(slope, slope_err),
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10},
        fontsize=13)
plt.legend()
plt.savefig('figures//f55_abb_4.pdf',format='pdf')
#plt.show()
plt.close()




########################
######Teil 2############
########################

Time, H2O_ppm, H2O_ppm_sd, O18_del, O18_del_sd, D_del, D_del_sd, O17_del, O17_del_sd= np.genfromtxt('data/teil2.txt', delimiter=",", skip_header=1265, skip_footer=700, usecols=(0,1,2,3,4,5,6,7,8), unpack=True)

print(len(H2O_ppm))
time=np.arange(0,len(H2O_ppm))*10.035/60

plt.errorbar(time, H2O_ppm, yerr=H2O_ppm_sd, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeit [min]', fontsize=13)
plt.ylabel('$H_2O$ [ppm]', fontsize=13)
plt.title('Abb. [6]: Verdunstender Wassertropfen $H_2 O$ [ppm]', fontsize=16)
plt.savefig('figures//f55_abb_6.pdf',format='pdf')
#plt.show()
plt.close()

plt.errorbar(time, O18_del/1000, yerr=O18_del_sd/1000, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeit [min]', fontsize=13)
plt.ylabel('$\\delta^{18} O$', fontsize=13)
plt.title('Abb. [7]: Verdunstender Wassertropfen $\\delta^{18} O$', fontsize=16)
plt.savefig('figures//f55_abb_7.pdf',format='pdf')
#plt.show()
plt.close()

plt.errorbar(time, D_del/1000, yerr=D_del_sd/1000, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeit [min]', fontsize=13)
plt.ylabel('$\\delta D$', fontsize=13)
plt.title('Abb. [8]: Verdunstender Wassertropfen $\\delta D$', fontsize=16)
plt.savefig('figures//f55_abb_8.pdf',format='pdf')
#plt.show()
plt.close()

plt.errorbar(time, O17_del/1000, yerr=O17_del_sd/1000, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeit [min]', fontsize=13)
plt.ylabel('$\\delta^{17} O$', fontsize=13)
plt.title('Abb. [9]: Verdunstender Wassertropfen $\\delta^{17} O$', fontsize=16)
plt.savefig('figures//f55_abb_9.pdf',format='pdf')
#plt.show()
plt.close()

#############
#### Teil 3 ####
##############

Time, D_del, D_del_sd, O18_del, O18_del_sd, O17_del, O17_del_sd= np.genfromtxt('data/rainfall_daily2.txt', delimiter='\t', skip_header=1, usecols=(0,3,4,5,6,7,8), unpack=True)


D_del=D_del/1000
D_del_sd=D_del_sd/1000
O18_del=O18_del/1000
O18_del_sd=O18_del_sd/1000
O17_del=O17_del/1000
O17_del_sd=O17_del_sd/1000

Time=np.append(Time, [240,241])
D_del=np.append(D_del, [rain293[0],rain298[0]])
D_del_sd=np.append(D_del_sd, [rain293[1],rain298[1]])
O18_del=np.append(O18_del, [rain293[2],rain298[2]])
O18_del_sd=np.append(O18_del_sd, [rain293[3],rain298[3]])
O17_del=np.append(O17_del, [rain293[4],rain298[4]])
O17_del_sd=np.append(O17_del_sd, [rain293[5],rain298[5]])


plt.errorbar(Time, D_del, yerr=D_del_sd, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeitraum 2016-2018', fontsize=13)
plt.ylabel('$\\delta D$', fontsize=13)
plt.title('Abb. [10]: Tägliche $\\delta D$ Werte', fontsize=16)
plt.savefig('figures//f55_abb_10.pdf',format='pdf')
plt.show()
plt.close()

popt, pcov = curve_fit(linear, Delta_O18, Delta_D, absolute_sigma=True, sigma=Delta_D_err)
plt.errorbar(O18_del, D_del, xerr=O18_del_sd, yerr=D_del_sd, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('$\\delta^{18} O$', fontsize=13)
plt.ylabel('$\\delta D$', fontsize=13)
plt.title('Abb. [11]: Waterline täglich', fontsize=16)
plt.savefig('figures//f55_abb_11.pdf',format='pdf')
plt.show()
plt.close()

D_del, D_del_sd, O18_del, O18_del_sd, O17_del, O17_del_sd= np.genfromtxt('data/rainfall_monthly2.txt', delimiter='\t', skip_header=1, usecols=(3,4,6,7,9,10), unpack=True)

D_del=D_del/1000
D_del_sd=D_del_sd/1000
O18_del=O18_del/1000
O18_del_sd=O18_del_sd/1000
O17_del=O17_del/1000
O17_del_sd=O17_del_sd/1000

D_del=np.append(D_del, [rain58[0]])
D_del_sd=np.append(D_del_sd, [rain58[1]])
O18_del=np.append(O18_del, [rain58[2]])
O18_del_sd=np.append(O18_del_sd, [rain58[3]])
O17_del=np.append(O17_del, [rain58[4]])
O17_del_sd=np.append(O17_del_sd, [rain58[5]])


Time=np.arange(0,len(D_del))
plt.errorbar(Time, D_del, yerr=D_del_sd, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('Zeitraum 2016-2018', fontsize=13)
plt.ylabel('$\\delta D$', fontsize=13)
plt.title('Abb. [12]: Monatliche $\\delta D$ Werte', fontsize=16)
plt.savefig('figures//f55_abb_12.pdf',format='pdf')
plt.show()
plt.close()

plt.errorbar(O18_del, D_del, xerr=O18_del_sd, yerr=D_del_sd, fmt='.', linewidth=1, linestyle='', color='black')
plt.xlabel('$\\delta^{18} O$', fontsize=13)
plt.ylabel('$\\delta D$', fontsize=13)
plt.title('Abb. [13]: Waterline monatlich', fontsize=16)
plt.savefig('figures//f55_abb_13.pdf',format='pdf')
plt.show()
plt.close()

