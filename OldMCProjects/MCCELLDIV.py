import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
import sys

TMAX=30_000
M=16
MAX=89
# multipliers=(-0.583 , .266,.0376,-4.66,-.04)
multipliers =(-0.5845411 , 0.12693621, 0.04277835, -4.66577823, -0.03983156)

@jit(nopython=True)
def calcComb(NA, la, lA,lc,multipliers):
    halpha, ha, ka, hc, kac=multipliers
    if NA == 0 or lA == 0 or (NA == lA):
        comb = (halpha*la + ha*lA + ka*la*lA + hc*lc + kac*lA*lc)
    else:
        # comb = (NA * np.log(NA) - lA * np.log(lA) - (NA - lA) * np.log(NA - lA))\
        #            + (halpha*la + ha*lA + ka*la*lA + hc*lc + kac*lA*lc)\
        #     +.5*np.log(2*math.pi*NA)-.5*np.log(2*math.pi*lA)-.5*np.log(2*math.pi*(NA-lA))
        comb =(NA*np.log(NA)) - (lA*np.log(lA)) - ((NA-lA)*np.log(NA-lA)) + 0.5*(np.log(NA) - np.log(lA) - np.log(NA - lA) - np.log(2*np.pi))+ (halpha * la + ha * lA + ka * la * lA + hc * lc + kac * lA * lc)

    return comb


@jit(nopython=True)
def calceitherPij(multipliers):

    pijdiv = np.zeros((MAX+1,MAX+1),np.float64)
    pijnodiv = np.zeros((MAX+1,MAX+1),np.float64)

    for i in range(MAX+1):
        Q = calcQ(i,multipliers)

        for j in range(MAX+1):
            for lA in range(0, i + 1):
                la = j - lA
                if (la>=0 and la <= M):
                        #print(la,lA,j, (1/Q)*np.exp(calcComb(i,la,lA,0,multipliers)))

                    pijdiv[i][j] += (1/Q)*np.exp(calcComb(i,la,lA,1,multipliers))
                    pijnodiv[i][j] += (1/Q)*np.exp(calcComb(i,la,lA,0,multipliers))


        Ppostdiv = np.zeros_like(pijdiv)
        for m in range(MAX+1):
            postnorm = np.sum(pijdiv[m,:])
            Ppostdiv[m,:] =pijdiv[m,:]/postnorm

        normalization = np.sum(pijdiv[i,:])+np.sum(pijnodiv[i,:])
        pijdiv[i][:]=pijdiv[i][:]/normalization
        pijnodiv[i][:]=pijnodiv[i][:]/normalization

    return pijdiv,pijnodiv,Ppostdiv

@jit(nopython=True)
def calcQ(NA,multipliers):
    Q = 0
    for i in range(0, M + 1):  ##lalpha
        for j in range(0, NA + 1):  # la
            for l in range(0,2): #lc

                Q += np.exp(calcComb(NA,i,j,l,multipliers))

    return Q

# pijdiv, pijnodiv,pijpostdiv = calceitherPij(multipliers)
# # pijnodiv = pijnodiv[1:,1:]
# print(pijdiv[25])
# pijnodiv = pijnodiv[1:,1:]
# pijpostdiv = pijpostdiv[1:,1:]


