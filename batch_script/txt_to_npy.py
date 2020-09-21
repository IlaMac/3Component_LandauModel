import numpy as np
import sys
import os



beta_low=float(sys.argv[1])
beta_high=float(sys.argv[2])
nbeta=int(sys.argv[3])
h=float(sys.argv[4])
e=sys.argv[5]

beta=np.zeros((nbeta))

if( (h).is_integer()): h=int(h)

L=np.array([8, 10])


for l in range(len(L)):
    BASEDIR=("/home/ilaria/Desktop/MultiComponents_SC/Output_2C/L%d_e%s_h%s_bmin%s_bmax%s" %(L[l], e,  h, beta_low, beta_high))
    print(BASEDIR)
    for b in range(nbeta):
        
        beta[b]=beta_low +b*((beta_high-beta_low)/(nbeta-1))
       
        file_DS=("%s/beta_%d/Dual_Stiffness" %(BASEDIR, b))
        print(file_DS)
        Ds=np.loadtxt(file_DS + ".txt", usecols=0, unpack=True)
        np.save(file_DS + ".npy", Ds)

        file_M=("%s/beta_%d/Magnetization" %(BASEDIR, b))
        M=np.loadtxt(file_M + ".txt", usecols=0, unpack=True)
        np.save(file_M + ".npy", M)
        
        file_E=("%s/beta_%d/Energy" %(BASEDIR, b))
        E=np.loadtxt(file_E + ".txt", usecols=0, unpack=True)
        np.save(file_E + ".npy", E)

        file_Psi=("%s/beta_%d/Psi_density" %(BASEDIR, b))
        Psi=np.loadtxt(file_Psi + ".txt", usecols=(0,1))
        np.save(file_Psi + ".npy", Psi)

        ###############################
        #  Remove the old .txt files  #
        ###############################

        os.remove(file_DS + ".txt")
        os.remove(file_M + ".txt")
        os.remove(file_E + ".txt")
        os.remove(file_Psi + ".txt")
