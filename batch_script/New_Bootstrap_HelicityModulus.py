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
eta=(sys.argv[8])
if( (nu).is_integer()): nu=int(nu)
if( (e).is_integer()): e=int(e)
#if( (h).is_integer()): h=int(h)


L=[]
for ind in range(9, len(sys.argv)):
    L.append(int(sys.argv[ind]))

#if( (h).is_integer()): h=int(h)

#beta_low=1.46
#beta_high=1.55
#nbeta=64
#e=0
#h=1
#nu=0
#
#L=([8, 10, 12, 16, 20])
#
#folder_out=("/Users/ilaria/Desktop/MultiComponents_SC/Output_3C/e_%s/nu_%s/h_%s" %(e, nu, h))
#

beta=np.zeros((nbeta))


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{bm}')


color=iter(plt.cm.rainbow(np.linspace(0,1,len(L)+1)))

NC=3 #number of components
h3=float(h)**3

fig, (ax1) = plt.subplots(nrows=1, figsize=(8,8))
ax1.set_ylabel(r"$L\Upsilon$")
ax1.set_xlabel(r"$\beta$")

for l in range(len(L)):
    
    N=L[l]**3
    c_m=next(color)

    Helicity_sum_mean=np.zeros((nbeta))
    Helicity_sum_err=np.zeros((nbeta))
    Helicity_sum_err_2=np.zeros((nbeta))
    
    Helicity_single_mean=np.zeros((NC, nbeta))
    Helicity_single_err=np.zeros((NC, nbeta))

    Helicity_mixed_mean=np.zeros((NC, nbeta))
    Helicity_mixed_err=np.zeros((NC, nbeta))

    D2H_Dd2_mean=np.zeros((NC, nbeta))
    DH_Dd_mean=np.zeros((NC, nbeta))
    DH_Dd_2_mean=np.zeros((NC, nbeta))
    DH_Ddij_mean=np.zeros((NC, nbeta))
    D2H_Dd2ij_mean=np.zeros((NC, nbeta))

    D2H_Dd2_err=np.zeros((NC, nbeta))
    DH_Dd_err=np.zeros((NC, nbeta))
    DH_Dd_2_err=np.zeros((NC, nbeta))
    DH_Ddij_err=np.zeros((NC, nbeta))
    D2H_Dd2ij_err=np.zeros((NC, nbeta))

    N_dataset=100

    BASEDIR=("%s/L%d_rho1_eta%s_e%s_h%s_nu%s_bmin%s_bmax%s"  %(folder_out, L[l], eta, e,  h, nu, beta_low, beta_high))
    print(BASEDIR)
    data_tau_max=np.loadtxt("%s/tau_max.txt" %BASEDIR, dtype=np.str)
    tau_max=np.amax(np.array(data_tau_max[1], dtype=np.float))
    
    data_transient_time=np.loadtxt("%s/transient_time.txt" %BASEDIR, dtype=np.str)
    transient_time=int(np.amax(np.array(data_transient_time[1], dtype=np.float)))

    print(L[l], tau_max, transient_time)

    for b in range(nbeta):
        beta[b]=beta_low +b*(beta_high -beta_low)/(nbeta-1)

        file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
            
    #####################################################################################
    #                                                                                   #
    #                              Single component Helicity                            #
    #                                                                                   #
    #####################################################################################

        DH_Ddi=np.asarray(file['Measurements']['DH_Ddi'])
        D2H_Dd2i=np.asarray(file['Measurements']['D2H_Dd2i'])

        for alpha in range(NC):
            DH_Dd=h3*DH_Ddi[:,alpha]
            D2H_Dd2=h3*D2H_Dd2i[:,alpha]

            #cut of the transient regime
            DH_Dd=DH_Dd[transient_time:]
            D2H_Dd2=D2H_Dd2[transient_time:]
            
            Nblocks=100
            block_size=int(len(DH_Dd)/Nblocks)
            while((block_size<(20*tau_max)) and (Nblocks>20) ):
                Nblocks=int(Nblocks*0.5)
                block_size=int(len(DH_Dd)/Nblocks)
            DH_Dd_block=np.zeros((Nblocks, block_size))
            D2H_Dd2_block=np.zeros((Nblocks, block_size))
            for block in range(Nblocks):
                DH_Dd_block[block]=DH_Dd[block*block_size: (block+1)*block_size]
                D2H_Dd2_block[block]=D2H_Dd2[block*block_size: (block+1)*block_size]

            meanDH_Dd_resampling=np.zeros((N_dataset))
            stdDH_Dd_resampling=np.zeros((N_dataset))
            
            meanDH_Dd_2_resampling=np.zeros((N_dataset))
            stdDH_Dd_2_resampling=np.zeros((N_dataset))
            
            meanD2H_Dd2_resampling=np.zeros((N_dataset))
            stdD2H_Dd2_resampling=np.zeros((N_dataset))

            #bootstrap resampling extract Nblocks with replacement and form a new set of data from which compute Cv, E_err, Cv_err
            for n in range(N_dataset):
                resampling= np.random.choice(Nblocks,Nblocks)

                DH_Dd_resampling=DH_Dd_block[resampling]
                DH_Dd_2_resampling=DH_Dd_block[resampling]*DH_Dd_block[resampling]
                D2H_Dd2_resampling=D2H_Dd2_block[resampling]

                meanDH_Dd_resampling[n]=np.mean(DH_Dd_resampling)
                stdDH_Dd_resampling[n]=np.sqrt(np.var(DH_Dd_resampling)/(Nblocks -1))
                
                meanDH_Dd_2_resampling[n]=np.mean(DH_Dd_2_resampling)
                stdDH_Dd_2_resampling[n]=np.sqrt(np.var(DH_Dd_2_resampling)/(Nblocks -1))

                meanD2H_Dd2_resampling[n]=np.mean(D2H_Dd2_resampling)
                stdD2H_Dd2_resampling[n]=np.sqrt(np.var(D2H_Dd2_resampling)/(Nblocks -1))
          
            DH_Dd_mean[alpha, b]=np.mean(meanDH_Dd_resampling)
            DH_Dd_err[alpha, b]=np.mean(stdDH_Dd_resampling)

            DH_Dd_2_mean[alpha, b]=np.mean(meanDH_Dd_2_resampling)
            DH_Dd_2_err[alpha, b]=np.mean(stdDH_Dd_2_resampling)

            D2H_Dd2_mean[alpha, b]=np.mean(meanD2H_Dd2_resampling)
            D2H_Dd2_err[alpha, b]=np.mean(stdD2H_Dd2_resampling)

            #For the moment I consider DH_Dd_mean to be just zero as it should be at the equilibrium
            Helicity_single_mean[alpha, b]= 1./(N*h3) *(D2H_Dd2_mean[alpha, b] - (beta[b])*(DH_Dd_2_mean[alpha, b] - (DH_Dd_mean[alpha, b]**2) ) )
            Helicity_single_err[alpha, b]= 1./(N*h3) *np.sqrt(D2H_Dd2_err[alpha, b]**2 + (beta[b]*DH_Dd_2_err[alpha, b])**2 + (2*beta[b]*DH_Dd_err[alpha, b])**2 )

    #####################################################################################
    #                                                                                   #
    #                               Mixed component Helicity                            #
    #                                                                                   #
    #####################################################################################

        D2H_Ddidj=np.asarray(file['Measurements']['D2H_Dd2ij'])
        DH_Ddi_DH_Ddj_temp=[]
        for alpha in range(NC):
            for gamma in range(alpha+1, NC):
                DH_Ddi_DH_Ddj_temp.append(DH_Ddi[:,alpha]*DH_Ddi[:,gamma])
        
        DH_Ddi_DH_Ddj_temp=np.asarray(DH_Ddi_DH_Ddj_temp)
        for alpha in range(NC):
            
            D2H_Dd2ij=h3*D2H_Ddidj[:,alpha]
            DH_Ddi_DH_Ddj=h3*h3*DH_Ddi_DH_Ddj_temp[alpha,:]

            #cut of the transient regime
            DH_Ddi_DH_Ddj=DH_Ddi_DH_Ddj[transient_time:]
            D2H_Dd2ij=D2H_Dd2ij[transient_time:]

            D2H_Dd2ij_block=np.zeros((Nblocks, block_size))
            DH_Ddi_DH_Ddj_block=np.zeros((Nblocks, block_size))

            for block in range(Nblocks):
                D2H_Dd2ij_block[block]=D2H_Dd2ij[block*block_size: (block+1)*block_size]
                DH_Ddi_DH_Ddj_block[block]=DH_Ddi_DH_Ddj[block*block_size: (block+1)*block_size]

            meanD2H_Dd2ij_resampling=np.zeros((N_dataset))
            stdD2H_Dd2ij_resampling=np.zeros((N_dataset))

            meanDH_Ddi_DH_Ddj_resampling=np.zeros((N_dataset))
            stdDH_Ddi_DH_Ddj_resampling=np.zeros((N_dataset))
            #bootstrap resampling extract Nblocks with replacement and form a new set of data from which compute Cv, E_err, Cv_err
            for n in range(N_dataset):
                resampling= np.random.choice(Nblocks,Nblocks)
                meanD2H_Dd2ij_resampling[n]=np.mean(D2H_Dd2ij_block[resampling])
                stdD2H_Dd2_resampling[n]=np.sqrt(np.var(D2H_Dd2ij_block[resampling])/(Nblocks -1))
                meanDH_Ddi_DH_Ddj_resampling[n]=np.mean(DH_Ddi_DH_Ddj_block[resampling])
                stdDH_Ddi_DH_Ddj_resampling[n]=np.sqrt(np.var(DH_Ddi_DH_Ddj_block[resampling])/(Nblocks -1))

            D2H_Dd2ij_mean[alpha, b]=np.mean(meanD2H_Dd2ij_resampling)
            D2H_Dd2ij_err[alpha, b]=np.mean(stdD2H_Dd2ij_resampling)

            DH_Ddij_mean[alpha, b]=np.mean(meanDH_Ddi_DH_Ddj_resampling)
            DH_Ddij_err[alpha, b]=np.mean(stdDH_Ddi_DH_Ddj_resampling)

            Helicity_mixed_mean[alpha, b]=1./(N*h3) *( D2H_Dd2ij_mean[alpha, b] - (beta[b]*DH_Ddij_mean[alpha, b]))
            Helicity_mixed_err[alpha, b]= 1./(N*h3) *np.sqrt( D2H_Dd2ij_err[alpha, b]**2 + (beta[b]*DH_Ddij_err[alpha,b])**2 )

        Helicity_sum_mean[b]+= Helicity_single_mean[alpha,b] +2*Helicity_mixed_mean[alpha, b]
        Helicity_sum_err_2[b]+=( Helicity_single_err[alpha,b]**2 + 4*Helicity_mixed_err[alpha, b]**2)

        Helicity_sum_err[b]=np.sqrt(Helicity_sum_err_2[b])

    data_helicity=np.array([beta, Helicity_sum_mean, Helicity_sum_err])
    data_helicity=np.transpose(data_helicity)
    np.savetxt("%s/Helicity_modulus_sum.txt" %(BASEDIR), data_helicity, fmt=['%lf','%lf', '%lf'])

    data_helicity_singlec=np.array([beta, Helicity_single_mean[0,:], Helicity_single_err[0,:], Helicity_single_mean[1,:], Helicity_single_err[1,:], Helicity_single_mean[2,:], Helicity_single_err[2,:]])
    data_helicity_singlec=np.transpose(data_helicity_singlec)
    np.savetxt("%s/Helicity_modulus_singlecomponents.txt" %(BASEDIR), data_helicity_singlec, fmt=['%lf','%lf', '%lf','%lf', '%lf','%lf', '%lf'])

    data_helicity_mixedc=np.array([beta, Helicity_mixed_mean[0,:], Helicity_mixed_err[0,:], Helicity_mixed_mean[1,:], Helicity_mixed_err[1,:], Helicity_mixed_mean[2,:], Helicity_mixed_err[2,:]])
    data_helicity_mixedc=np.transpose(data_helicity_mixedc)
    np.savetxt("%s/Helicity_modulus_mixedcomponents.txt" %(BASEDIR), data_helicity_mixedc, fmt=['%lf','%lf', '%lf','%lf', '%lf','%lf', '%lf'])

    data_currents=np.array([beta, DH_Dd_mean[0,:], DH_Dd_err[0,:], DH_Dd_mean[1,:], DH_Dd_err[1,:], DH_Dd_mean[2,:], DH_Dd_err[2,:]])
    data_currents=np.transpose(data_currents)
    np.savetxt("%s/Currents.txt" %(BASEDIR), data_currents, fmt=['%lf','%lf', '%lf','%lf', '%lf','%lf', '%lf'])


    ax1.errorbar(beta, L[l]*float(h)*Helicity_sum_mean, yerr= Helicity_sum_err, capsize=2, c=c_m)

fig.savefig("%s/Helicity_modulus_sum_h%s_bmin%s_bmax%s.png" %(folder_out, h, beta_low, beta_high))
plt.show()


