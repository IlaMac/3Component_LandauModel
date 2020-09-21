import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
import math
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import scipy.integrate as integrate
import random
import h5py


folder_out=sys.argv[1]
beta_low=float(sys.argv[2])
beta_high=float(sys.argv[3])
nbeta=int(sys.argv[4])
e=float(sys.argv[5])
h=(sys.argv[6])
nu=float(sys.argv[7])
if( (nu).is_integer()): nu=int(nu)
if( (e).is_integer()): e=int(e)
#if( (h).is_integer()): h=int(h)


L=[]
for ind in range(8, len(sys.argv)):
    L.append(int(sys.argv[ind]))

beta=np.zeros((nbeta))


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{bm}')
fig, ax1 = plt.subplots(nrows=2, figsize=(4,8))
ax1[0].set_title(r"$h=%s$; $e=%s$; $\nu=%s$" %(h, e, nu))
ax1[0].set_xlabel(r"$\beta$")
ax1[0].set_ylabel(r"$E/V$")
ax1[1].set_xlabel(r"$\beta$")
ax1[1].set_ylabel(r"$C_{v}$")

color=iter(plt.cm.rainbow(np.linspace(0,1,len(L)+1)))


for l in range(len(L)):

    Cv_mean=np.zeros((nbeta))
    Cv_err=np.zeros((nbeta))
    E_mean=np.zeros((nbeta))
    E_err=np.zeros((nbeta))

    c_m=next(color)

    N_dataset=100

    BASEDIR=("%s/L%d_rho1_eta1_e%s_h%s_nu%s_bmin%s_bmax%s"  %(folder_out, L[l], e,  h, nu, beta_low, beta_high))
    data_tau_max=np.loadtxt("%s/tau_max.txt" %BASEDIR, dtype=np.str)
    tau_max=np.amax(np.array(data_tau_max[1], dtype=np.float))

    data_transient_time=np.loadtxt("%s/transient_time.txt" %BASEDIR, dtype=np.str)
    transient_time=int(np.amax(np.array(data_transient_time[1], dtype=np.float)))

    for b in range(nbeta):
        beta[b]=beta_low +b*(beta_high -beta_low)/(nbeta-1)
        
        file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
        E=np.asarray(file['Measurements']['E'])

        #cut of the transient regime:
        E=E[transient_time:]
      
        #split the N measurements in Nblocks blocks according to the autocorrelation time tau
     
        Nblocks=100
        block_size=int(len(E)/Nblocks)
        while((block_size<(20*tau_max)) and (Nblocks>20) ):
            Nblocks=int(Nblocks*0.5)
            block_size=int(len(E)/Nblocks)
        #block_size=int(20*tau_max)

        E_block=np.zeros((Nblocks, block_size))
        for block in range(Nblocks):
            E_block[block]=E[block*block_size: (block+1)*block_size]
       
        meanE_resampling=np.zeros((N_dataset))
        varE_resampling=np.zeros((N_dataset))
        stdE_resampling=np.zeros((N_dataset))
        #bootstrap resampling extract Nblocks with replacement and form a new set of data from which compute Cv, E_err, Cv_err
        for n in range(N_dataset):
            resampling= np.random.choice(Nblocks,Nblocks)
            E_resampling=E_block[resampling]
            meanE_resampling[n]=np.mean(E_resampling/(L[l]**3))
            stdE_resampling[n]=np.sqrt(np.var(E_resampling/(L[l]**3))/(Nblocks -1))
            varE_resampling[n]=np.var(E_resampling)

        E_mean[b]=np.mean(meanE_resampling)
        E_err[b]=np.mean(stdE_resampling)
        Cv_mean[b]=(beta[b]/(L[l]**3))*np.mean(varE_resampling)
        Cv_err[b]=(beta[b]/(L[l]**3))*np.sqrt(np.var(varE_resampling)/(N_dataset-1))

    data_energy=np.vstack((beta, E_mean, E_err))
    np.savetxt("%s/Energy.txt" %(BASEDIR), data_energy)
    np.savetxt("%s/Specific_Heat.txt" %(BASEDIR), (beta, Cv_mean, Cv_err))


    ax1[0].plot(beta, E_mean, '-', c=c_m)
    #for block in range(nblocks):
    #    ax1[0].plot(beta, meanE_resampling[block,:], '-')
    ax1[0].errorbar(beta, E_mean, yerr=E_err, capsize=2, c=c_m, label="L=%s" %L[l])
    ax1[1].plot(beta, Cv_mean, '-', c=c_m)
    ax1[1].errorbar(beta, Cv_mean, yerr=Cv_err, c=c_m, capsize=2)

ax1[0].legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig("%s/Energy_h%s_bmin%s_bmax%s.png" %(folder_out, h, beta_low, beta_high))



