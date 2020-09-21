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

beta_low=0.5
beta_high=5.0
nbeta=32
e=0.
h=2
nu=1.
L=[8,10, 12]
#, 16, 20]

#beta_low=float(sys.argv[1])
#beta_high=float(sys.argv[2])
#nbeta=int(sys.argv[3])
#e=float(sys.argv[4])
#h=float(sys.argv[5])
#nu=float(sys.argv[6])
if( (nu).is_integer()): nu=int(nu)
if( (e).is_integer()): e=int(e)

folder_out=("/Users/ilaria/Desktop/MultiComponents_SC/Output_3C/e_%s/nu_%s/h_%s" %(e, nu, h))
transient=0.5

#L=[]
#for ind in range(7, len(sys.argv)):
#    L.append(int(sys.argv[ind]))


beta=np.zeros((nbeta))

color=iter(plt.cm.rainbow(np.linspace(0,1,len(L)+1)))


plt.rc('text.latex', preamble=r'\usepackage{bm}')


fig_all, axs = plt.subplots(nrows=3, ncols=2, figsize=(8,8))
fig_all.suptitle(r"FAST CHECK $h=%s$; $e=%s$; $\nu=%s$" %(h, e, nu))
axs[0,0].set_xlabel(r"$\beta$")
axs[0,0].set_ylabel(r"$E$")
axs[0,1].set_xlabel(r"$\beta$")
axs[0,1].set_ylabel(r"$C_v$")
axs[1,0].set_xlabel(r"$\beta$")
axs[1,0].set_ylabel(r"$M$")
axs[1,1].set_xlabel(r"$\beta$")
axs[1,1].set_ylabel(r"$U$")
axs[2,0].set_xlabel(r"$\beta$")
axs[2,0].set_ylabel(r"$L\rho$")
axs[2,1].set_xlabel(r"$\beta$")
axs[2,1].set_ylabel(r"$L\Upsilon$")

NC=3

for l in range(len(L)):
	N=L[l]**3
    
	U_mean=np.zeros((nbeta))
	M_mean=np.zeros((nbeta))
	
	U1_mean=np.zeros((nbeta))
	M1_mean=np.zeros((nbeta))
	
	U2_mean=np.zeros((nbeta))
	M2_mean=np.zeros((nbeta))

	E_mean=np.zeros((nbeta))
	Cv_mean=np.zeros((nbeta))
	
	E1_mean=np.zeros((nbeta))
	Cv1_mean=np.zeros((nbeta))
	
	E2_mean=np.zeros((nbeta))
	Cv2_mean=np.zeros((nbeta))

	Ds_mean=np.zeros((nbeta))
	
	Ds1_mean=np.zeros((nbeta))
	
	Ds2_mean=np.zeros((nbeta))

	Helicity_sum_mean=np.zeros((nbeta))
	
	Helicity_sum_mean1=np.zeros((nbeta))
	
	Helicity_sum_mean2=np.zeros((nbeta))

	Helicity_single_mean=np.zeros((NC, nbeta))
	Helicity_mixed_mean=np.zeros((NC, nbeta))
	
	Helicity_single_mean1=np.zeros((NC, nbeta))
	Helicity_mixed_mean1=np.zeros((NC, nbeta))
	
	Helicity_single_mean2=np.zeros((NC, nbeta))
	Helicity_mixed_mean2=np.zeros((NC, nbeta))


	D2H_Dd2_mean=np.zeros((NC, nbeta))
	DH_Dd_mean=np.zeros((NC, nbeta))
	DH_Dd_2_mean=np.zeros((NC, nbeta))
	DH_Ddij_mean=np.zeros((NC, nbeta))
	D2H_Dd2ij_mean=np.zeros((NC, nbeta))
	
	D2H_Dd2_mean1=np.zeros((NC, nbeta))
	DH_Dd_mean1=np.zeros((NC, nbeta))
	DH_Dd_2_mean1=np.zeros((NC, nbeta))
	DH_Ddij_mean1=np.zeros((NC, nbeta))
	D2H_Dd2ij_mean1=np.zeros((NC, nbeta))
	
	D2H_Dd2_mean2=np.zeros((NC, nbeta))
	DH_Dd_mean2=np.zeros((NC, nbeta))
	DH_Dd_2_mean2=np.zeros((NC, nbeta))
	DH_Ddij_mean2=np.zeros((NC, nbeta))
	D2H_Dd2ij_mean2=np.zeros((NC, nbeta))

	c_m=next(color)
	BASEDIR=("%s/L%d_a0_b1_eta1_e%s_h%s_nu%s_bmin%s_bmax%s"  %(folder_out, L[l], e,  h, nu, beta_low, beta_high))
    
	for b in range(nbeta):
		beta[b]=beta_low +b*(beta_high -beta_low)/(nbeta-1)

#####################################################################################
#                                                                                   #
#                              			Energy	    		                        #
#                                                                                   #
#####################################################################################

		file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
		E=np.asarray(file['Measurements']['E'])

		print(L[l], len(E))
        #cut of the approximated transient regime:
		E=E[int(transient*len(E)):]

		E_mean[b]=np.mean(E)
		Cv_mean[b]=(beta[b]/(L[l]**3))*np.var(E)
		
		E1=E[int(0.5*len(E)):]
		E1_mean[b]=np.mean(E1)
		Cv1_mean[b]=(beta[b]/(L[l]**3))*np.var(E1)
		
		E2=E[:int(0.5*len(E))]
		E2_mean[b]=np.mean(E2)
		Cv2_mean[b]=(beta[b]/(L[l]**3))*np.var(E2)

#####################################################################################
#                                                                                   #
#                              		Magnetization    		                        #
#                                                                                   #
#####################################################################################

		file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
		M=np.asarray(file['Measurements']['m'])

        #cut of the approximated transient regime:
		M=M[int(transient*len(M)):]

		M_mean[b]=np.mean(M)
		U_mean[b]=np.mean(np.power(M,4))/(3*np.power(np.mean(np.power(M,2)),2))
		
		M1=M[int(0.5*len(M)):]
		M1_mean[b]=np.mean(M1)
		U1_mean[b]=np.mean(np.power(M1,4))/(3*np.power(np.mean(np.power(M1,2)),2))

		M2=M[:int(0.5*len(M))]
		M2_mean[b]=np.mean(M2)
		U2_mean[b]=np.mean(np.power(M2,4))/(3*np.power(np.mean(np.power(M2,2)),2))

#####################################################################################
#                                                                                   #
#                              	Dual Stiffness	    		                        #
#                                                                                   #
#####################################################################################

		file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
		Ds=np.asarray(file['Measurements']['ds'])
		factor=(h)**2

        #cut of the approximated transient regime:
		Ds=factor*Ds[int(transient*len(Ds)):]

		Ds_mean[b]=L[l]*np.mean(Ds)


#####################################################################################
#                                                                                   #
#                              	Helicity modulus    		                        #
#                                                                                   #
#####################################################################################


		file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
		print('%s/beta_%d/Output.h5' %(BASEDIR, b))
	            
	        
		DH_Ddi=np.asarray(file['Measurements']['DH_Ddi'])
		D2H_Dd2i=np.asarray(file['Measurements']['D2H_Dd2i'])
		D2H_Ddidj=np.asarray(file['Measurements']['D2H_Dd2ij'])
	
		DH_Ddi_DH_Ddj_temp=[]
		for alpha in range(NC):
			for gamma in range(alpha+1, NC):
				DH_Ddi_DH_Ddj_temp.append(DH_Ddi[:,alpha]*DH_Ddi[:,gamma])
	        
		DH_Ddi_DH_Ddj_temp=np.asarray(DH_Ddi_DH_Ddj_temp)
	
		h3=h**3
		
		for alpha in range(NC):
	
			DH_Dd=h3*DH_Ddi[:,alpha]
			D2H_Dd2=h3*D2H_Dd2i[:,alpha]
			D2H_Dd2ij=h3*D2H_Ddidj[:,alpha]
			
			DH_Ddi_DH_Ddj=h3*h3*DH_Ddi_DH_Ddj_temp[alpha,:]
	        
	        #cut of the transient regime
			DH_Dd=DH_Dd[int(transient*len(DH_Dd)):]
			D2H_D2d=D2H_Dd2[int(transient*len(D2H_Dd2)):]
			D2H_Dd2ij=D2H_Dd2ij[int(transient*len(D2H_Dd2ij)):]
			DH_Ddi_DH_Ddj=DH_Ddi_DH_Ddj[int(transient*len(DH_Ddi_DH_Ddj)):]
	
	
			D2H_Dd2_mean[alpha,b]=np.mean(D2H_Dd2)
			DH_Dd_2_mean[alpha,b]=np.mean(DH_Dd*DH_Dd)
			D2H_Dd2ij_mean[alpha,b]=np.mean(D2H_Dd2ij)
			DH_Ddij_mean[alpha,b]=np.mean(DH_Ddi_DH_Ddj)
	
	
			Helicity_single_mean[alpha,b]= 1./(N*h3) *(D2H_Dd2_mean[alpha, b] - (beta[b])*(DH_Dd_2_mean[alpha, b]) )
			Helicity_mixed_mean[alpha,b]=1./(N*h3)*( D2H_Dd2ij_mean[alpha, b] - (beta[b]*DH_Ddij_mean[alpha, b]))
	
			Helicity_sum_mean[b]+= (Helicity_single_mean[alpha,b] +2*Helicity_mixed_mean[alpha, b])
	
		print(b, D2H_Dd2_mean[0,b], DH_Dd_2_mean[0,b] )


	axs[0,0].plot(beta, E_mean/N, '-', c=c_m, label="%d" %L[l])
	axs[0,1].plot(beta, Cv_mean, '-', c=c_m)
	axs[0,0].plot(beta, E1_mean/N, '--', c=c_m)
	axs[0,1].plot(beta, Cv1_mean, '--', c=c_m)
	axs[0,0].plot(beta, E2_mean/N, '-.', c=c_m)
	axs[0,1].plot(beta, Cv2_mean, '-.', c=c_m)

	axs[1,0].plot(beta, M_mean, '-', c=c_m)
	axs[1,1].plot(beta, U_mean, '-', c=c_m)
	axs[1,0].plot(beta, M1_mean, '--', c=c_m)
	axs[1,1].plot(beta, U1_mean, '--', c=c_m)
	axs[1,0].plot(beta, M2_mean, '-.', c=c_m)
	axs[1,1].plot(beta, U2_mean, '-.', c=c_m)

	axs[2,0].plot(beta, L[l]*float(h)*Ds_mean, '-', c=c_m)
	axs[2,1].plot(beta, L[l]*float(h)*Helicity_sum_mean, '-', c=c_m)

	lines=[]
	labels=[]
	for ax in fig_all.axes:
		axLine, axLabel = ax.get_legend_handles_labels()
		lines.extend(axLine)
		labels.extend(axLabel)

fig_all.tight_layout()
fig_all.subplots_adjust(bottom=0.2, top=0.95)
fig_all.legend(lines, labels, loc = 'lower center', title="L", ncol=len(L))
fig_all.savefig("%s/Fast_check_bmin%s_bmax%s.png" %(folder_out, beta_low, beta_high))
plt.show()
