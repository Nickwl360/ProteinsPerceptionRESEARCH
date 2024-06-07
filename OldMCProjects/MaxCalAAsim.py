import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
# constants
Tmax = 30000
M = 15
dt = 1
MAX = 75
@jit(nopython=True)
def calcComb(NA, la, lb,halpha,ha,ka):
    if NA == 0 or lb == 0 or (NA == lb):
        comb = (halpha * la + ha * lb + ka * la * lb)
    else:
        comb = (NA * np.log(NA) - lb * np.log(lb) - (NA - lb) * np.log(NA - lb))\
                   + (halpha * la + ha * lb + ka * (la * lb)) +.5*np.log(2*math.pi*NA)-.5*np.log(2*math.pi*lb)-.5*np.log(2*math.pi*(NA-lb))
    return comb

@jit(nopython=True)
def calcComb2(NA,la,lb,halpha,ha,ka):
    if NA == 0 or lb == 0 or (NA == lb):
        comb = (halpha * la + ha * lb + ka * la*NA)
    else:
        comb = (NA * np.log(NA) - lb * np.log(lb) - (NA - lb) * np.log(NA - lb))\
                   + (halpha * la + ha * lb + ka * (NA * la)) +.5*np.log(2*math.pi*NA)-.5*np.log(2*math.pi*lb)-.5*np.log(2*math.pi*(NA-lb))
    return comb
@jit(nopython=True)
def calcComb3(Ni,Nj,la,lb,halpha,ha,ka,km):
    if Nj == 0 or lb == 0 or (Nj == lb):
        comb = (halpha * la + ha * lb + ka * la * lb + km * la * Ni)
    else:
        comb = (Nj * np.log(Nj) - lb * np.log(lb) - (Nj - lb) * np.log(Nj - lb)) \
               + (halpha * la + ha * lb + ka * lb * la + km*la*Ni) + .5 * np.log(2 * math.pi * Nj) - .5 * np.log(2 * math.pi * lb) - .5 * np.log(2 * math.pi * (Nj - lb))
    return comb

@jit(nopython=True)
def calcPij(halpha,ha,ka):
    Pij = np.zeros((MAX, MAX), dtype=np.float64)
    for i in range(0,MAX):
        Q= calcQ(i, halpha, ha, ka)
        for j in range(0,MAX):
            ploop = 0
            for lb in range(0, i + 1):
                la = j - lb
                if 0 <= la <= M:
                    ploop += (1/Q)*np.exp(calcComb(i,la,lb,halpha,ha,ka))

            Pij[i][j] = ploop
    return Pij

@jit(nopython=True)
def calcPijk(halpha,ha,ka,km):
    Pij = np.zeros((MAX, MAX ,MAX), dtype=np.float64)
    for i in range(0,MAX):
        for j in range(0,MAX):
            Q = calcQ2(i, j, halpha, ha,ka,km)
            for k in range(0,MAX):
                ploop = 0
                for lb in range(0, j + 1):
                    la = k - lb
                    if 0 <= la <= M:
                        ploop += (1/Q)*np.exp(calcComb3(i,j,la,lb,halpha,ha,ka,km))

                Pij[i][j][k] = ploop
    return Pij
@jit(nopython=True)
def calcQ(NA,halpha,ha,ka):
    Q = 0
    for i in range(0, M + 1):  ##lalpha
        for j in range(0, NA + 1):  # la
            Q += np.exp(calcComb(NA,i,j,halpha,ha,ka))
    return Q
@jit(nopython=True)
def calcQ2(Ni,Nj,halpha,ha,ka,km):
    Q = 0
    for i in range(0, M + 1):  ##lalpha
        for j in range(0, Nj + 1):  # la
            Q += np.exp(calcComb3(Ni,Nj, i, j, halpha, ha, ka,km))
    return Q
def nextNA(NA, Parr):
    probs = Parr[NA, :]
    randnum = np.random.random()
    SUM = 0
    NAj = 0
    for i in range(len(probs)):
        SUM += probs[i]
        if SUM > randnum:
            NAj = i
            SUM = -100

    return NAj
def nextNAMem(Ni,Nj, Parr):
    probs = Parr[Ni,Nj, :]
    randnum = np.random.random()
    SUM = 0
    NAj = 0
    for i in range(len(probs)):
        SUM += probs[i]
        if SUM > randnum:
            NAj = i
            SUM = -100

    return NAj
def simulation(params):
    halpha,ha,ka = params
    NA = 1
    t = 0
    A = [1]
    pij = calcPij(halpha,ha,ka)
    while t < Tmax:
        NA = nextNA(NA, pij)
        t += dt
        A.append(NA)

    return A
def simulation2(params):
    halpha,ha,ka,km = params
    NAi = 1
    t = 0
    A = [NAi]
    pijk=calcPijk(halpha,ha,ka,km)
    while t < Tmax:
        if t ==0:
            NA = nextNAMem(NAi, A[t], pijk)
        else:
            NA = nextNAMem(A[t-1],A[t], pijk)

        t += dt
        A.append(NA)

    return A

normalAA= (-.512,.585,.0298)
MemoryAAperterbation=(-.512,.585,.0298,.001)
MemAA1eback= (-0.511674236,	0.580576876,	0.030100914,	0.000798134)

# #
#data = simulation2(MemAA1eback) #for compare
# data = simulation2(MemoryAAperterbation)
# #np.save('Mem10Milk1e-3trial1.npy',data)
# ts = np.linspace(0, Tmax, len(data))
# plt.plot(ts, data)
# plt.xlabel('time (s)')
# plt.ylabel('Number of gene A')
# plt.show()
