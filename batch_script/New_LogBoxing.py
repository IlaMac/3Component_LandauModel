import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
import math
import h5py


beta_low=float(sys.argv[1])
beta_high=float(sys.argv[2])
nbeta=int(sys.argv[3])
BASEDIR=sys.argv[4]
L=sys.argv[5]
nu=float(sys.argv[6])
e=float(sys.argv[7])

beta=np.zeros((nbeta))

##this are the tag used in writing the h5 file
#Observables=["E", "m", "ds"]

if(e>0):
        Observables=np.array(["E", "m", "ds"])
else:
    if(nu>0):
        Observables=np.array(["E", "m", "D2H_Dd2i", "DH_Ddi", "D2H_Dd2ij"])
    else:
        Observables=np.array(["E", "m", "D2H_Dd2i", "DH_Ddi"])


transient_max=np.zeros((len(Observables)))


for name in range(len(Observables)):
    A_name=Observables[name]
    transient_list=np.zeros((nbeta))
    for b in range(nbeta):
        beta[b]=beta_low +b*(beta_high -beta_low)/(nbeta-1)
        base=2
        exp=2
        box_length=base**exp
        file=h5py.File('%s/beta_%d/Output.h5' %(BASEDIR, b), 'r')
        A=np.asarray(file['Measurements']['%s' %(Observables[name])])
        if((Observables[name]=="D2H_Dd2i") or (Observables[name]=="DH_Ddi") or (Observables[name]=="D2H_Dd2ij")):
            A=A[:,0]
        tot_length=box_length
        start=0
        A_mean=[]
        A_std=[]
        bins=[]
        while (tot_length< len(A)):
            A_mean.append(np.mean(A[start:tot_length]))
            A_std.append(np.sqrt(np.var(A[start:tot_length])/(len(A[start:tot_length]) -1)))
            bins.append(tot_length)
            start=tot_length
            exp+=1
            box_length=base**exp
            tot_length+=box_length
        A_mean=np.array(A_mean)
        A_std=np.array(A_std)
        bins=np.array(bins)
        #plt.errorbar(bins, A_mean, yerr=A_std, fmt='o-', label= "%s L=%s" %(Observables[name], L))
        #plt.legend(loc="best")
        #plt.show()

        ##### Find where the plateau starts by taking the minimum values of the derivative of the binned funciton ####
        A_diff=np.diff(A_mean)
        index=np.argmin(np.abs(A_diff))
        #### The information to be extracted is the time up to the end of the bin where the plateau is observed: bin[index] ####            
        transient_list[b]=bins[index]
        file_Aout=("%s/beta_%d/Thermalization_%s.txt" %(BASEDIR, b, A_name))
        np.savetxt(file_Aout, np.transpose( [bins, A_mean, A_std]), fmt='%19.12e', delimiter=' ',  newline=os.linesep)
    transient_max[name]=np.amax(transient_list)

data=np.vstack((Observables, transient_max))
np.savetxt("%s/transient_time.txt" %BASEDIR, data, fmt="%s")


