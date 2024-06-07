import numpy as np
import matplotlib.pyplot as plt 
import operator as op
from functools import reduce
import sys 
from scipy.optimize import minimize
import time
from numba import jit, cuda, njit
import math


"""
edited 8/12/23
to run use:
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1

and modules:
module load cuda11.0/toolkit/11.0.3
module load compilers/anaconda-2021.11

Ted Kaczynski 
"""



@cuda.jit
def get_micro(p,lalphs,las,lbetas,lbs,halpha,ha,Kaalpha,Kabeta,Na,Nb):#

    def log_binomial(n,k):#log binomial of stirling
        def logstir(a):#stiriling approx
            return .5*math.log(2*np.pi*a)+(a*math.log(a/np.e))

        if n==k:
            return(0)
        if k==0:
            return(0)
        else:
            return logstir(n)-(logstir(k)+logstir(n-k))	

    #p=np.zeros((M+1,Na+1,M+1,Nb+1))#prob distribution (creation A, surviving A, Creation B, surviving B) 

    for lalpha in lalphs:
        for la in las:
            for lbeta in lbetas:
                for lb in lbs:
                    val=log_binomial(Na,la)+log_binomial(Nb,lb)
                    expval=(halpha*(lalpha+lbeta))+(ha*(la+lb))+(Kaalpha*((lalpha*la)+(lbeta*lb)))+(Kabeta*((lbeta*la)+(lalpha*lb)))
                    p[lalpha,la,lbeta,lb]=val+expval

                    





    #p=np.exp(p-np.max(p))
    


# Create the data array - usually initialized some other way
M=15
Na=30
Nb=40

lalphs=np.arange(0,M+1,1)
las=np.arange(0,Na+1,1)
lbetas=np.arange(0,M+1,1)
lbs=np.arange(0,Nb+1,1)




p=np.zeros((M+1,Na+1,M+1,Nb+1))

# Set the number of threads in a block
threadsperblock = 256 

# Calculate the number of thread blocks in the grid
blockspergrid = (p.size + (threadsperblock - 1)) // threadsperblock

# Now start the kernel
tik=time.perf_counter()
get_micro[blockspergrid, threadsperblock](p,lalphs,las,lbetas,lbs,0.295,1.526,-0.034,-0.244,Na,Nb)
tok=time.perf_counter()
# Print the result
print("done")





print("time ",tok-tik)



