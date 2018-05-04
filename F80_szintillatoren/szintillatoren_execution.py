'''
Created on 20.04.2018

@author: Nils Schmitt
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2


####################
#Plots der Versuchsdurchführung#
####################


#Coarse Gain 4
raw = np.loadtxt('data//verstaerkung_4_t62_raw.dat', 
                 delimiter = '\t', unpack=True)
a = raw[:1025]
plt.plot(a, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [3]: Coarse Gain 4', fontsize=16)
plt.savefig('figures//f80_abb_3.pdf',format='pdf')
#plt.show()
plt.close()


#Coarse Gain 64
raw = np.loadtxt('data//verstaerkung_64_t60_raw.dat', 
                 delimiter = '\t', unpack=True)
a = raw[0:1025]
plt.plot(a, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [4]: Coarse Gain 64', fontsize=16)
plt.savefig('figures//f80_abb_4.pdf',format='pdf')
#plt.show()
plt.close()


#Coarse Gaun 128
raw = np.loadtxt('data//verstaerkung_128_t62_raw.dat', 
                delimiter = '\t', unpack=True)
a = raw[0:1025]
plt.plot(a, linestyle='-', marker='.',
         color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [5]: Coarse Gain 128', fontsize=16)
plt.text(850,300,'%s'%('Photopeak'),
         fontsize=13)
plt.arrow(930, 290, 40, -70, shape='full', width=3, 
          length_includes_head=True, color='black')
plt.text(650,120,'%s'%('Compton-Effekt'),
         fontsize=13)
plt.arrow(790, 110, 50, -40, shape='full', width=3, 
          length_includes_head=True, color='black')
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f80_abb_5.pdf',format='pdf')
#plt.show()
plt.close()


#Szintillatoren entfernt
raw = np.loadtxt('data//parallel_raw.dat', 
                 delimiter = '\t', unpack=True)
a = raw[0:1025]
plt.plot(a, linestyle='-', marker='.',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [7]: Szintillatoren voneinander entfernt', fontsize=16)
plt.savefig('figures//f80_abb_7.pdf',format='pdf')
#plt.show()
plt.close()


#Szintillatoren antiparallel
raw = np.loadtxt('data//antiparallel_raw.dat', 
                 delimiter = '\t', unpack=True)             
a = raw[0:1025]
plt.plot(a, marker='.', linestyle='-',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [8]: Szintillatoren antiparallel', fontsize=16)
plt.savefig('figures//f80_abb_8.pdf',format='pdf')
#plt.show()
plt.close()


#Spannungsabhängigkeit 2000V
raw = np.loadtxt('data//myon_t60_u2000_raw.dat',
                 delimiter = '\t', unpack=True)
back = np.loadtxt('data//myon_t60_u2000_back_raw.dat',
                 delimiter = '\t', unpack=True)            
a = raw[0:1025]-back[0:1025]
plt.plot(a, marker='.', linestyle='-',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [9]: Arbeitsspannung U=-2000V', fontsize=16)
plt.savefig('figures//f80_abb_9.pdf',format='pdf')
#plt.show()
plt.close()


#Spannungsabhängigkeit 1850V
raw = np.loadtxt('data//myon_t60_u1850_raw.dat',
                 delimiter = '\t', unpack=True)
back = np.loadtxt('data//myon_t63_u1850_back_raw.dat',
                 delimiter = '\t', unpack=True)            
a = raw[0:1025]-back[0:1025]
plt.plot(a, marker='.', linestyle='-',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [10]: Arbeitsspannung U=-1850V', fontsize=16)
plt.savefig('figures//f80_abb_10.pdf',format='pdf')
#plt.show()
plt.close()


#Spannungsabhängigkeit 1700V
raw = np.loadtxt('data//myon_t62_u1700_raw.dat',
                 delimiter = '\t', unpack=True)
back = np.loadtxt('data//myon_t62_u1700_back_raw.dat',
                 delimiter = '\t', unpack=True)            
a = raw[0:1025]-back[0:1025]
plt.plot(a, marker='.', linestyle='-',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [11]: Arbeitsspannung U=-1700V', fontsize=16)
plt.savefig('figures//f80_abb_11.pdf',format='pdf')
#plt.show()
plt.close()


#Spannungsabhängigkeit 1500V
raw = np.loadtxt('data//myon_t61_u1500_raw.dat',
                 delimiter = '\t', unpack=True)
back = np.loadtxt('data//myon_t72_u1500_back_raw.dat',
                 delimiter = '\t', unpack=True)            
a = raw[0:1025]-back[0:1025]
plt.plot(a, marker='.', linestyle='-',
            color='black', label='Messdaten')
plt.xlabel('Channel', fontsize=13)
plt.ylabel('Counts', fontsize=13)
plt.title('Abb. [12]: Arbeitsspannung U=-1500V', fontsize=16)
plt.savefig('figures//f80_abb_12.pdf',format='pdf')
#plt.show()
plt.close()


#