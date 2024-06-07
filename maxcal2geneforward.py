import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

import cProfile

# constants
Tmax = 200000
halpha = 0.259
ha = 1.526
kaa = -0.034
kab = -0.244
M = 31
dt = 300
comb_cache = defaultdict(list)
MAX = 40


def newcomb(N, L):
    if N == 0 or L == 0 or N == L:
        comb_cache[(N, L)] = 0
    else:
        comb = (N * np.log(N) - L * np.log(L) - (N - L) * np.log(N - L)) \
               + .5 * np.log(2 * math.pi * N) - .5 * np.log(
            2 * math.pi * L) - .5 * np.log(2 * math.pi * (N - L))

        comb_cache[(N, L)] = comb
    return
def calcallcomb():
    for i in range(0, MAX):
        for j in range(0, i + 1):
            newcomb(i, j)
    return
def calcCombs(NA, NB, la, lA, lb, lB, halpha, ha, kaa, kab):
    e = (halpha * (la + lb) + ha * (lA + lB) + kaa * (la * lA + lb * lB) + kab * (lb * lA + la * lB))
    return comb_cache[(NA, lA)] + comb_cache[(NB, lB)] + e

def calcPlabs(NA, NB, halpha, ha, kaa, kab):
    Plabs = np.zeros((M + 1, NA + 1, M + 1, NB + 1), dtype=np.float64)

    combs = []
    Q = 0
    # lalpha-x la-y
    for i in range(0, M + 1):
        for j in range(0, NA + 1):
            for k in range(0, M + 1):
                for l in range(NB + 1):
                    comb = math.exp(calcCombs(NA, NB, i, j, k, l, halpha, ha, kaa, kab))
                    combs.append(comb)

    for index, value in enumerate(combs):
        Q += value

    for i in range(0, M + 1):
        for j in range(0, NA + 1):
            for k in range(0, M + 1):
                for l in range(NB + 1):
                    index = i * (NA + 1) * (M + 1) * (NB + 1) + j * (M + 1) * (NB + 1) + k * (NB + 1) + l
                    Plabs[i][j][k][l] = (1 / Q) * combs[index]

    return Plabs
rng = np.random.default_rng(100)


def nextNs(NA, NB, Parr):
    pSUM = np.zeros((MAX,MAX),dtype=np.float64)
    for j in range(MAX):
        for l in range(MAX):
            loop1 = 0
            for lA in range(NA+1):
                loop4 = 0
                for lB in range(NB+1):
                    la = j - lA
                    lb = l - lB
                    if 0<=la<32 and 0<=lb<32:
                        loop4 += Parr[la][lA][lb][lB]

                loop1 += loop4
            pSUM[j][l] = loop1

    randnum = rng.random()
    SUM2 =0
    NAj = 0
    NBl = 0
    for i in range(MAX):
        for j in range(MAX):
            SUM2 += pSUM[i][j]
            if SUM2 > randnum:
                NAj = i
                NBl = j
                SUM2 -= 100

    return NAj, NBl


def simulation():
    NA = 0
    NB = 0
    t = 0
    A = [0]
    B = [0]
    calcallcomb()
    while t < Tmax:
        newprob= calcPlabs(NA,NB,halpha,ha,kaa,kab)
        NA, NB = nextNs(NA, NB, newprob)
        t += dt
        A.append(NA)
        B.append(NB)
        print(t)


    return A, B


# cProfile.run('calcPijkl(halpha,ha,kaa,kab)')
As,Bs = simulation()
ts = np.linspace(0, Tmax, len(As))
plt.plot(ts, As, c='r')
plt.plot(ts, Bs, c='b')
plt.xlabel('time (s)')
plt.ylabel('Number')
plt.title('A (red), B(blue) vs time')
plt.show()
