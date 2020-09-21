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

color=iter(plt.cm.rainbow(np.linspace(0,1,len(L)+1)))


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{bm}')
fig, ax1 = plt.subplots(nrows=2, figsize=(4,8))
ax1[0].set_title(r"$h=%s$; $e=%s$; $\nu=%s$" %(h, e, nu))
ax1[0].set_xlabel(r"$\beta$")
ax1[0].set_ylabel(r"$M$")
ax1[1].set_xlabel(r"$\beta$")
ax1[1].set_ylabel(r"$U$")

for l in range(len(L)):

    U_mean=np.zeros((nbeta))
    U_err=np.zeros((nbeta))
    M_mean=np.zeros((nbeta))
    M_err=np.zeros((nbeta))

    c_m=next(color)
    BASEDIR=("%s/L%d_rho1_eta1_e%s_h%s_nu%s_bmin%s_bmax%s"  %(folder_out, L[l], e,  h, nu, beta_low, beta_high))
    
    data_tau_max=np.loadtxt("%s/tau_max.txt" %BASEDIR, dtype=np.str)
    tau_max=np.amax(np.array(data_tau_max[1], dtype=np.float))

    data_transient_time=np.loadtxt("%s/transient_time.txt" %BASEDIR, dtype=np.str)
    transient_time=int(np.amax(np.array(data_transient_time[1], dtype=np.float)))

    N_dataset=100

    for b in range(nbeta):
        beta[b]=beta_low +b*(beta_high -beta_low)/(nbeta-1)

        file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
        M=np.asarray(file['Measurements']['m'])

        #cut of the transient regime:
        M=M[transient_time:]
       
        #split the N measurements in Nb blocks according to the autocorrelation time tau
        Nblocks=100
        block_size=int(len(M)/Nblocks)
        while((block_size<(20*tau_max)) and (Nblocks>20) ):
            Nblocks=int(Nblocks*0.5)
            block_size=int(len(M)/Nblocks)
        M_block=np.zeros((Nblocks, block_size))
        for block in range(Nblocks):
            M_block[block]=M[block*block_size: (block+1)*block_size]

        meanM_resampling=np.zeros((N_dataset))
        stdM_resampling=np.zeros((N_dataset))
        U_resampling=np.zeros((N_dataset))

        #bootstrap resampling extract M blocks with replacement and form a new set of data from which compute Cv, E_err, Cv_err
        for n in range(N_dataset):
            resampling= np.random.choice(Nblocks,Nblocks)
            M_resampling=M_block[resampling]
            meanM_resampling[n]=np.mean(M_resampling)
            stdM_resampling[n]=np.sqrt(np.var(M_resampling)/(Nblocks -1))
            U_resampling[n]=np.mean(np.power(M_resampling,4))/(3*np.power(np.mean(np.power(M_resampling,2)),2))

        M_mean[b]=np.mean(meanM_resampling)
        M_err[b]=np.mean(stdM_resampling)
        U_mean[b]=np.mean(U_resampling)
        U_err[b]=np.sqrt(np.var(U_resampling)/(N_dataset-1))

    np.savetxt("%s/Magnetization.txt" %(BASEDIR), (beta, M_mean, M_err))
    np.savetxt("%s/Binder_cumulant.txt" %(BASEDIR), (beta, U_mean, U_err))

    ax1[0].plot(beta, M_mean, '-', c=c_m)
    ax1[0].errorbar(beta, M_mean, yerr=M_err, capsize=2, c=c_m,label="L=%s" %L[l])
    ax1[1].plot(beta, U_mean, '-', c=c_m)
    ax1[1].errorbar(beta, U_mean, yerr=U_err, c=c_m, capsize=2)

ax1[0].legend(loc="best")
plt.tight_layout()
plt.savefig("%s/Magnetization_h%s_bmin%s_bmax%s.png" %(folder_out, h, beta_low, beta_high))


