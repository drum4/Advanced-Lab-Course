'''
Created on Sep 29, 2018

@author: schmi
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy import *

rc('font',**{'family':'serif','serif':['Linux Libertine O']})
plt.rcParams['errorbar.capsize']=2





####################
### Saugvermögen ###
####################
PA = 1013
vol1 = 0.04*np.ones(5)*10**(-3)
vol1_err = np.sqrt(0.0003**2+0.001**2)*10**(-3)
t1 = np.array([122.8,57.1,24.7,12.6,8.7])
t1_err = np.array([0.3,0.1,0.1,0.2,0.3])
p1 = np.array([5.1e-6,1e-5,2.5e-5,5e-5,7.5e-5])
p1_err = np.array([0.1e-6,0.1e-5,0.1e-5,0.1e-5,0.1e-5])
kap1 = vol1/t1*PA/p1
kap1_err = vol1/t1**2*PA/p1*t1_err

vol2 = 0.1*np.ones(3)*10**(-3)
vol2_err = np.sqrt(0.0006**2+0.002**2)*10**(-3)
t2 = np.array([20.6,15.6,6.2])
t2_err = np.array([0.1,0.2,0.3])
p2 = np.array([7.5e-5,1e-4,2.5e-4])
p2_err = np.array([0.1e-5,0.1e-4,0.1e-4])
kap2 = vol2/t2*PA/p2
kap2_err = vol2/t2**2*PA/p2*t2_err

vol5 = 0.25*np.ones(2)*10**(-3)
vol5_err = np.sqrt(0.003**2+0.02**2)*10**(-3)
t5 = np.array([15.6,7.6])
t5_err = np.array([0.2,0.3])
p5 = np.array([2.5e-4,5e-4])
p5_err = np.array([0.1e-4,0.1e-4])
kap5 = vol5/t5*PA/p5
kap5_err = vol5/t5**2*PA/p5*t5_err

vol20 = 1.2*np.ones(4)*10**(-3)
vol20_err = np.sqrt(0.01**2+0.02**2)*10**(-3)
t20 = np.array([37.5,24.5,18.2,6.2])
t20_err = np.array([.1,.1,.1,.3])
p20 = np.array([5e-4,7.5e-4,1e-3,2.5e-3])
p20_err = np.array([.1e-4,.1e-4,.1e-3,.1e-3])
kap20 = vol20/t20*PA/p20
kap20_err = vol20/t20**2*PA/p20*t20_err

volK = 20*np.ones(5)*10**(-3)
volK_err = .3e-3
tK = np.array([118.7,61.2,42.1,31.6,15.8])
tK_err = .1*np.ones(5)
pK = np.array([2.5e-3,5e-3,7.5e-3,1.1e-2,2.7e-2])
pK_err = np.array([.1e-3,.1e-3,.1e-3,.1e-2,.1e-2])
kapK = volK/tK*PA/pK
kapK_err = volK/tK**2*PA/pK*tK_err

plt.errorbar(p1, kap1, xerr=p1_err, yerr=kap1_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='0.1ml Kapillar')
plt.errorbar(p2, kap2, xerr=p2_err, yerr=kap2_err,
             fmt='.', linewidth=1,
             linestyle='', color='red',
             label='0.2ml Kapillar')
plt.errorbar(p5, kap5, xerr=p5_err, yerr=kap5_err,
             fmt='.', linewidth=1,
             linestyle='', color='blue',
             label='0.5ml Kapillar')
plt.errorbar(p20, kap20, xerr=p20_err, yerr=kap20_err,
             fmt='.', linewidth=1,
             linestyle='', color='green',
             label='2ml Kapillar')
plt.errorbar(pK, kapK, xerr=pK_err, yerr=kapK_err,
             fmt='.', linewidth=1,
             linestyle='', color='magenta',
             label='Kolben')
plt.xscale('log')
plt.xlabel('Druck [mbar]', fontsize=13)
plt.ylabel('Effektives Saugvermögen [l/s]', fontsize=13)
plt.title('Fig. [1]: Saugvermögen in Abhängigkeit vom Druck', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f70_abb_1.pdf',format='pdf')
#plt.show()
plt.close()

kap = np.concatenate((kap1[:-1],kap2[:-1],kap5[:-1],kap20[:-1],kapK))
S = np.mean(kap[2:-2])
S_err = np.std(kap[2:-2])
print('S =',S,'+/-',S_err)









##################
#Teilchen gegen Druck gegen freie Weglänge#
##################

t = np.arange(1e-6, 1e3)
data1 = 10**18*t
data2 = 10**-5/t
fig, ax1 = plt.subplots()
plt.xscale('log')
plt.yscale('log')
color = 'tab:red'
ax1.set_xscale('log')
ax1.set_xlabel('Druck [mbar]', fontsize=13)
ax1.set_ylabel('Teilchendichte [1/m$^3$]', color=color, fontsize=13)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Freie Weglänge [cm]', color=color, fontsize=13)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
plt.yscale('log')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#ax1.title.set_text('Abb. [2]: Teilchendichte und freie Weglänge', fontsize=16)
plt.suptitle('Abb. [2]: Teilchendichte und freie Weglänge [Nitrogen]', fontsize=16)
fig.subplots_adjust(top=.92) #adjust title position
plt.savefig('figures//f70_abb_2.pdf',format='pdf')
#plt.show()
plt.close()









##################
#Leitwerte Rohr#
##################
p_ur = np.array([9.7e-6,1e-5,1.1e-5,1.5e-5,2.4e-5,3.3e-5,4.1e-5,9.2e-5,1.9e-4,3.4e-4,6.6e-4,2.9e-3,9.8e-3,2.5e-2,7.4e-2])
p_ur_err = np.array([.1e-6,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-4,.1e-4,.1e-4,.1e-3,.1e-3,.1e-2,.1e-2])
p_or = np.array([4.9e-4,7.5e-4,1.1e-3,2.5e-3,4.8e-3,7.6e-3,1e-2,2.5e-2,5e-2,7.6e-2,.1,.25,.5,.76,1.1])
p_or_err = np.array([.1e-4,.1e-4,.1e-3,.1e-3,.1e-3,.1e-3,.1e-2,.1e-2,.1e-2,.1e-2,.1e-1,.1e-1,.1e-1,.1e-1,.1])
l_r = S*p_ur/(p_or-p_ur)
l_r_err = np.sqrt((S_err*p_ur/(p_or-p_ur))**2+(S*p_ur*p_or_err/(p_or-p_ur)**2)**2+((S/(p_or-p_ur)+S*p_ur/(p_or-p_ur)**2)*p_ur_err)**2)

plt.errorbar(p_ur, l_r, xerr=p_ur_err, yerr=l_r_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.xscale('log')
plt.xlabel('Druck [mbar]', fontsize=13)
plt.ylabel('Leitwert [l/s]', fontsize=13)
plt.title('Abb. [3]: Leitwert Rohr', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f70_abb_3.pdf',format='pdf')
#plt.show()
plt.close()
ll_r = np.mean(l_r[1:-4])
ll_r_err = np.std(l_r[1:-4])
print('L_R =',ll_r,'+/-',ll_r_err)









##################
#Leitwerte Blende#
##################
p_ub = np.array([1.1e-5,1.1e-5,1.3e-5,1.8e-5,2.1e-5,2.5e-5,4.4e-5,8.3e-5,1.2e-4,1.5e-4,3.6e-4,7.7e-4,1.2e-3,1.8e-3,5.6e-3,1.3e-2,2.2e-2])
p_ub_err = np.array([.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-4,.1e-4,.1e-4,.1e-4,.1e-3,.1e-3,.1e-3,.1e-2,.1e-2])
p_ob = np.array([8.5e-5,9.6e-5,2.4e-4,5.7e-4,7.8e-4,1.1e-3,2.4e-3,5.1e-3,7.6e-3,1e-2,2.6e-2,5.2e-2,7.5e-2,9.7e-2,.25,.5,.76])
p_ob_err = np.array([.1e-5,.1e-5,.1e-4,.1e-4,1e-4,.1e-3,.1e-3,.1e-3,.1e-3,.1e-2,.1e-2,.1e-2,.1e-2,.1e-2,.1e-1,.1e-1,.1e-1])
l_b = S*p_ub/(p_ob-p_ub)
l_b_err = np.sqrt((S_err*p_ub/(p_ob-p_ub))**2+(S*p_ub*p_ob_err/(p_ob-p_ub)**2)**2+((S/(p_ob-p_ub)+S*p_ub/(p_ob-p_ub)**2)*p_ub_err)**2)

plt.errorbar(p_ub, l_b, xerr=p_ub_err, yerr=l_b_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.xscale('log')
plt.xlabel('Druck [mbar]', fontsize=13)
plt.ylabel('Leitwert [l/s]', fontsize=13)
plt.title('Abb. [4]: Leitwert Blende', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f70_abb_4.pdf',format='pdf')
#plt.show()
plt.close()
ll_b = np.mean(l_b[6:-4])
ll_b_err = np.std(l_b[6:-4])
print('L_B =',ll_b,'+/-',ll_b_err)









##################
#Leitwerte Beide#
##################
p_ug = np.array([1.2e-5,1.2e-5,1.3e-5,1.3e-5,1.8e-5,2.4e-5,3.1e-5,3.9e-5,8.3e-5,1.6e-4,2.6e-4,5e-4,1.9e-3,6.3e-3,1.3e-2,2.3e-2])
p_ug_err = np.array([.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-5,.1e-4,.1e-4,.1e-4,.1e-3,.1e-3,.1e-2,.1e-2])
p_og = np.array([4e-4,4.6e-4,7.4e-4,9.8e-4,2.5e-3,5e-3,7.6e-3,1e-2,2.6e-2,5e-2,7.4e-2,.1,.25,.5,.75,1])
p_og_err = np.array([.1e-4,.1e-4,.1e-4,.1e-4,.1e-3,.1e-3,.1e-3,.1e-2,.1e-2,.1e-2,.1e-2,.1e-1,.1e-1,.1e-1,.1e-1,.1])
l_g = S*p_ug/(p_og-p_ug)
l_g_err = np.sqrt((S_err*p_ug/(p_og-p_ug))**2+(S*p_ug*p_og_err/(p_og-p_ug)**2)**2+((S/(p_og-p_ug)+S*p_ug/(p_og-p_ug)**2)*p_ug_err)**2)

plt.errorbar(p_ug, l_g, xerr=p_ug_err, yerr=l_g_err,
             fmt='.', linewidth=1,
             linestyle='', color='black',
             label='Messpunkte mit Fehler')
plt.xscale('log')
plt.xlabel('Druck [mbar]', fontsize=13)
plt.ylabel('Leitwert [l/s]', fontsize=13)
plt.title('Abb. [5]: Leitwert Kombination', fontsize=16)
plt.legend(frameon=True, fontsize = 12)
plt.savefig('figures//f70_abb_5.pdf',format='pdf')
#plt.show()
plt.close()
ll_g = np.mean(l_g[4:-4])
ll_g_err = np.std(l_g[4:-4])
print('L_g =',ll_g,'+/-',ll_g_err)


#Kirchhoff
l_k = 1/(1/ll_b+1/ll_r)
l_k_err = l_k*np.sqrt((ll_b_err/ll_b)**2+(ll_r_err/ll_r)**2)
print('L_k =',l_k,'+/-',l_k_err)