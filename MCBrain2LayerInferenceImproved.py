import numpy as np
import matplotlib.pyplot as plt
import math
from MCBrain2layer import *
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import shgo
from scipy.sparse import lil_matrix
import scipy.io
# directory = 'JochNE25NR5Counts'
# if not os.path.exists(directory):
#     os.makedirs(directory)

# constants
# MAXTOP = 5
# MAXBOT = 12
MAXTOP = 6
MAXBOT = 26

#############JOCHEN DATA OLD TRAJS############
# jochdata = scipy.io.loadmat('StochasticRecurrentSymmetric (1).mat')
# print(jochdata.keys())
# Rkij = jochdata['R_kij']
# Ekij = jochdata['E_kij']
# set = 15
# Upleft=Rkij[0,:,set]
# Upright=Rkij[1,:,set]
# Botleft=Ekij[0,:,set]
# Botright=Ekij[1,:,set]
# dataa = np.round(Upleft[:]*4)
# datab = np.round(Upright[:]*4)
# datac = np.round(Botleft[:]*11)
# datad = np.round(Botright[:]*11)
set = 3
jochdatafix = scipy.io.loadmat('StochasticRecurrentSymmetricNE11NR4.mat')
#jochdatafix = scipy.io.loadmat('StochasticRecurrentSymmetricNE25NR5.mat')
#print(jochdatafix.keys())
Rkij = jochdatafix['R_kij']
Ekij = jochdatafix['E_kij']
times = jochdatafix['dt']
print(times)
dataa=Rkij[0,:,set]/(MAXTOP-1)
datab=Rkij[1,:,set]/(MAXTOP-1)
datac=Ekij[0,:,set]/(MAXBOT-1)
datad=Ekij[1,:,set]/(MAXBOT-1)


def countbrain(dataa,datab,datac,datad):
    count = np.zeros((MAXTOP, MAXTOP, MAXBOT, MAXBOT,MAXTOP, MAXTOP, MAXBOT, MAXBOT),dtype=np.uint8)
    for i in range(0, len(datab) - 1):
        print(i)
        count[int(dataa[i])][int(datab[i])][int(datac[i])][int(datad[i])][int(dataa[i+1])][int(datab[i + 1])][int(datac[i+1])][int(datad[i + 1])] += 1
    return count

def renormalize(Pijkl):
    normalizefactors = np.sum(Pijkl, axis=(4, 5, 6, 7))  # Sum over the end state indices
    normalizefactors[normalizefactors == 0] = 1.0
    Pijkl /= normalizefactors[:, :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    return Pijkl

def brainlikelyhood(params9, counts):
    #ULC,LLC,kcoop,kcomp,kdu,kud,kx = params7

    hgamma,hc,halpha,ha,kcoop,kcomp,kdu,kud,kx = params9

    epsilon2=0
    #(halpha, ha, hgamma, hc, kcoop, kcomp, kdu, kud, kx) = (-1 * ULC , ULC , -1 * LLC + epsilon2 / 2, LLC + -1 * epsilon2 / 2, kcoop, kcomp, kdu, kud, kx)
    params = (halpha, ha, halpha, ha , hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)

    Pmnop = runtime_program(params, Pmnop_prog)
    pmnopreshape = Pmnop.reshape((MAXTOP, MAXTOP, MAXBOT, MAXBOT, MAXTOP, MAXTOP, MAXBOT, MAXBOT))
    pmnonormal = renormalize(pmnopreshape)

    nonzero_counts = counts != 0
    nonzero_pmnonormal = pmnonormal != 0

    log_values = np.zeros_like(counts, dtype=float)
    log_values[nonzero_counts & nonzero_pmnonormal] = counts[nonzero_counts & nonzero_pmnonormal] * np.log(
        pmnonormal[nonzero_counts & nonzero_pmnonormal])
    L = np.sum(log_values)
    length=100_000_000
    #length=len(dataa)
    val = (-1*L)/length
    print('Likelyhood: ', val)
    return val

def minlikely(counts):
    #initial7=(.82,.3,2.05, 2.41, 0.81, 0.18, 0.44)
    #initial9strongdt01 = (-5.836,-6.09057,-3.8672,6.9068,2.05, 2.41, 0.81, 0.18, 0.44)
    #ADJUSTED hA TO MAX FROM DT = .01, adjusted Ks to other max
    #initial9strongdt001 = (-8.13844,-8.3985,-8.5233,11.50703375,  3.51690674,  1.55742667,  3.97934418,  0.02326176 , 0.90827147)
    #initial9weakdt001=(-6.58151137, -5.4149016 , -7.32583317, 11.50703375,  8.02477748,  7.15952577, 11.87505467,  0.31408652,  5.4198252)
    #initial9weakdt001=(-9.6, -7 , -8.5, 6.908, 4.62,  2.05, 6,  0.295,  1.6 )
    #initial9weak = (-7.507,-4.411,-3.8672,6.9068,2.05, 2.41, 0.81, 0.18, 0.44)
    #initialks = (2.05, 2.41, 0.81, 0.18, 0.44)
    #CalcedHsStrong= (-5.836,-6.09057,-3.8672,6.9068)   ##Hgamma, hC, halpha, hA
    #CalcedHsWeak= (-7.507,-4.411,-3.8672,6.9068)   ##Hgamma, hC, halpha, hA
    #bounds=((0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf))
    minimizer_kwargs = {"method": "SLSQP", "args": (counts)}
    minimizer_kwargsbasin = {"method": "COBYLA", "args": (counts,)}

    #SEED FROM NE11NR4 RESULTS
    initialdt001I0625=(-8.9673,-7.7323,-6.01935,-0.9932,4.722814,1.981144,6.05944,0.29747,1.53068)


    ###ALL REST
    maxL = minimize(brainlikelyhood, initialdt001I0625, args=(counts,), method='Powell' ,tol= 1e-9)

    #maxL = basinhopping(brainlikelyhood, initial7,minimizer_kwargs=minimizer_kwargsbasin )
    #maxL = shgo(brainlikelyhood, bounds=None,args=(counts,))

    maxparams = maxL.x
    return maxparams

###OG:          (-0.8245, 0.8245, -0.8245, 0.8245, -0.297, 0.297, -0.297, 0.297, 2.0, 2.43, 0.8175, 0.1681, 0.4359)

##JOCHTEST2,Set15, I =1
#[ 5.94159351e+00 , 6.65110825e-01,  7.68154925e+00 , 4.35942704e+00 , 1.06162505e+01 ,-3.85689956e-03,  3.26665828e+00]

if __name__ == "__main__":
    #count = countbrain(dataa, datab, datac, datad)

    #np.save('JochDt001I1Counts',count)
    #np.save('JochDt001I0625Counts',count)
    #np.save('JochDt001I375Counts',count)
    #np.save('JochDt001I6875Counts',count)
    #np.save('Jochdt001I0625NE25NR5counts',count,)
    #count= np.load('JochDt001I0625Counts.npy')
    #count=np.load('JochDt001I375Counts.npy')
    #count=np.load('JochDt001I6875Counts.npy')
    count = np.load('Jochdt001I0625NE25NR5counts.npy')
    total_zeros = np.count_nonzero(count)
    total_possible = (6 * 6 * 26 * 26) ** 2
    print(total_zeros)
    print(total_possible)
    #CalcedHsstrong= (-5.836,-6.09057,-3.8672,6.9068)
    #CalcedHsWeak= (-7.507,-4.411,-3.8672,6.9068)   ##Hgamma, hC, halpha, hA

    #params = minlikely(count)
    #print(params, 'max likelyhood: ', brainlikelyhood(params,count))
    #
    #epsilon1,epsilon2=0,0

    #Strong infered calcedhs dt01
    #paramsBEST= (-6.00432578 ,-6.03486149, -3.80626702 , 5.35061176 , 6.4340547  , 3.67099913 ,8.6066445  , 0.28647123 , 2.42905138)

    #weak infered calcedhas dt01
    #paramsBEST=(-6.58151137, -5.4149016 , -7.32583317, 11.50703375,  8.02477748,  7.15952577, 11.87505467,  0.31408652,  5.4198252 )

    #strong infered dt001BEST
    #paramsBEST = (-8.28640719, -8.42166917, -6.27641919, -0.61254556, 4.71567638, 2.15322098, 6.15779811, 0.29307981, 1.63738961)

    #print(brainlikelyhood(paramsBEST,count))



