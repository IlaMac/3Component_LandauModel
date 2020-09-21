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
import h5py

beta_low=float(sys.argv[1])
beta_high=float(sys.argv[2])
nbeta=int(sys.argv[3])
BASEDIR=sys.argv[4]
L=sys.argv[5]
nu=float(sys.argv[6])
e=float(sys.argv[7])

beta=np.zeros((nbeta))


if(e>0):
        Observables=np.array(["E", "m", "ds"])
else:
    if(nu>0):
        Observables=np.array(["E", "m", "D2H_Dd2i", "DH_Ddi", "D2H_Dd2ij"])
    else:
        Observables=np.array(["E", "m", "D2H_Dd2i", "DH_Ddi"])


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{bm}')

tau_max=np.zeros((len(Observables)))

for name in range(len(Observables)):
    tau=np.zeros((nbeta))
    for b in range(nbeta):
        beta[b]=beta_low +b*(beta_high -beta_low)/(nbeta-1)
        Obs_mean=np.zeros((nbeta))
        Obs_var=np.zeros((nbeta))
        file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
        Obs=np.asarray(file['Measurements']['%s' %(Observables[name])])
        if((Observables[name]=="D2H_Dd2i") or (Observables[name]=="DH_Ddi") or (Observables[name]=="D2H_Dd2ij")):
            Obs=Obs[:,0]
        A_Obs=acf(Obs, nlags=int(len(Obs)/10), fft=True)
        temp_tau=[]
        time_int=1000
        while(time_int<= len(A_Obs)):
            temp_tau=np.append(temp_tau, np.sum(A_Obs[:time_int]))
            time_int=time_int+1000
        tau[b]=np.amax(temp_tau)

    tau_max[name]=np.amax(tau)

data=np.vstack((Observables, tau_max))
np.savetxt("%s/tau_max.txt" %BASEDIR, data, fmt="%s")




