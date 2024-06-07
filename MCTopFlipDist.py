from MaxCalBrainTest1 import *
import numpy as np
import matplotlib.pyplot as plt

TMAX = 300000
samples = 10000

###FULLRESET
# initial=(0,24) #Tmax = 3000
# epsilon=1
# (halpha, ha, ka, kb) = (.0, .0, .15, .15)
# params = (halpha, ha, halpha - epsilon, ha + epsilon, ka, kb)
# Pijkl = (run_program(params, Pijkl_prog))
# Pijklreshape=Pijkl.reshape((MAX+1,MAX+1,MAX+1,MAX+1))
# Pijklnormal = renormalize(Pijklreshape)
# np.save('GammaDistP.npy',Pijklnormal)
# PGammaDist = np.load('GammaDistP.npy')
####RESET2
#initial=(0,24)   #Tmax=300000
# epsilon=.25
# initial = (0,24)
# (halpha, ha, ka, kb) = (0, 0.15, .1, .16)
# params = (halpha, ha, halpha - epsilon, ha + epsilon, ka, kb)
# Pijkl = (run_program(params, Pijkl_prog))
# Pijklreshape=Pijkl.reshape((MAX+1,MAX+1,MAX+1,MAX+1))
# Pijklnormal = renormalize(Pijklreshape)
# np.save('GammaDistPe2',Pijklnormal)
# PGammaDist = np.load('GammaDistPe2.npy')
###TRY3############
# initial=(0,24)
# epsilon=.12
# (halpha, ha, ka, kb) = (0.15, 0.15, .16, .1)
# params = (halpha, ha, halpha - epsilon, ha + epsilon, ka, kb)
# Pijkl = (run_program(params, Pijkl_prog))
# Pijklreshape=Pijkl.reshape((MAX,MAX,MAX,MAX))
# Pijklnormal = renormalize(Pijklreshape)
# np.save('GammaDistPe3',Pijklnormal)
# PGammaDist = np.load('GammaDistPe3.npy')
############4
# initial=(0,24)
# epsilon=.1
# (halpha, ha, ka, kb) = (0.1, 0.1, .1, .1)
# params = (halpha, ha, halpha - epsilon, ha + epsilon, ka, kb)
# Pijkl = (run_program(params, Pijkl_prog))
# Pijklreshape=Pijkl.reshape((MAX,MAX,MAX,MAX))
# Pijklnormal = renormalize(Pijklreshape)
# np.save('GammaDistPe4',Pijklnormal)
# PGammaDist = np.load('GammaDistPe4.npy')
###############5
# initial=(0,24)
# epsilon=1
# (halpha, ha, ka, kb) = (.8, .0, .3, .3)
# params = (halpha, ha, halpha - epsilon, ha + epsilon, ka, kb)
# Pijkl = (run_program(params, Pijkl_prog))
# Pijklreshape=Pijkl.reshape((MAX,MAX,MAX,MAX))
# Pijklnormal = renormalize(Pijklreshape)
# np.save('GammaDistPe5',Pijklnormal)
# PGammaDist = np.load('GammaDistPe5.npy')

def simtilflip(Nstart,Tmax,P):
    NA = Nstart[0]
    NB = Nstart[1]
    t = 0
    check =0
    flip = 0
    Pijklnormal=P
    while t < Tmax:
        NA, NB = nextNs(NA, NB, Pijklnormal)
        if NB<NA:
            check +=1
            if check >10:
                flip = t - 10
                break
        t += 1


    return flip

def sampletimes(initial,samples,P):
    fliptime=0
    i = 0
    toggletimes=[]
    while i < samples:
        toggletimes.append(simtilflip(initial,TMAX,P))
        i+=1
        print(i)
    return toggletimes
# ##V1###############
nbins=600
toggletimes =sampletimes(initial,samples,PGammaDist)
np.save('MCBrainflips.npy',toggletimes)
times = np.load("MCBrainflips.npy")
plt.hist(times,bins=nbins)
plt.show()
##################v2###############
# nbins=600
# toggletimes =sampletimes(initial,samples,PGammaDist)
# np.save('MCBrainflipsv2.npy',toggletimes)
# toggletimes = np.load("MCBrainflipsv2.npy")
# plt.hist(toggletimes,bins=nbins)
# plt.show()
###########plothist/save#############
# nbins=600
# toggletimes =sampletimes(initial,samples,PGammaDist)
# np.save('MCBrainflipsv5.npy',toggletimes)
# toggletimes = np.load("MCBrainflipsv5.npy")
# plt.hist(toggletimes,bins=nbins)
# plt.show()