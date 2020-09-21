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

beta_low=5.0
beta_high=20.0
nbeta=64
e=0.
h=0.2
nu=1.
L=[8, 10]

if( (nu).is_integer()): nu=int(nu)
if( (e).is_integer()): e=int(e)

folder_out=("/Users/ilaria/Desktop/MultiComponents_SC/Output_3C/e_%s/nu_%s" %(e, nu))
beta=np.zeros((nbeta))


for l in range(len(L)):
	N=L[l]**3

	BASEDIR=("%s/L%d_a0_b1_eta1_e%s_h%s_nu%s_bmin%s_bmax%s"  %(folder_out, L[l], e,  h, nu, beta_low, beta_high))
	folder_ranks=("%s/Ranks" %BASEDIR)
	try:
	    os.makedirs(folder_ranks)    
	    print("Directory " , folder_ranks ,  " Created ")
	except FileExistsError:
	    print("Directory " , folder_ranks ,  " already exists") 

	file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, nbeta-1), 'r')
	b_rank=np.asarray(file['Measurements']['rank'])
	
	ranks=np.zeros((nbeta, len(b_rank)))

	beta_rank=[]
	for b in range(nbeta):
		print(b)
		beta[b]=beta_low +b*(beta_high -beta_low)/(nbeta-1)

		file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
		b_rank=np.asarray(file['Measurements']['rank'])

		for r in range(len(b_rank)):
			ranks[b_rank[r], r]=b


	for r in range(nbeta):
		f, (ax1) = plt.subplots(ncols=1, figsize=(6,6) )
		f.suptitle(r"$rank=%s$" %(r))
		ax1.set_xlabel(r"$t_{MC}$")
		ax1.set_ylabel(r"$\beta$")
		ax1.plot(ranks[r, :])
		f.savefig("%s/rank_%s.png" %(folder_ranks, r))
		plt.close()





