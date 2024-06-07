import numpy as np
import matplotlib.pyplot as plt
import math
from time import perf_counter

# constants
Tmax = 600000
halpha = -.512
ha = .585
ka = .0298
M = 15
dt = 300

# variables

def calcProb(NA):
    Probs = np.zeros((M+1, NA+1))
                                    #lalpha-x la-y
    for i in range(0,M+1):

        for j in range(0,NA+1):
            Q = calcQ(NA)
            Probs[i][j]= (1/Q) * math.comb(NA,j)* math.exp(halpha*i+ha*j+ka*j*i)
    return Probs

def calcQ(NA):
    Q = 0
    for i in range(0,M+1):##lalpha
        jloop = 0
        for j in range(0,NA+1): #la
            jloop+= math.comb(NA,j) * math.exp(halpha*i+ha*j+ka*j*i)
        Q += jloop
    return Q

def nextNA(NA,Prob):
    probs = []
    for j in range(0,80):
        pSUM = 0
        for a in range(0,M+1):
            bloop = 0
            for b in range(0,NA+1):
                if (a+b-j) == 0:
                    bloop += Prob[a][b]
            pSUM += bloop
        probs.append(pSUM)
    randnum = np.random.random()
    SUM = 0
    NAj=0
    for i in range(len(probs)):
        SUM += probs[i]
        if SUM > randnum:
            NAj = i
            SUM = -100

    return NAj


def simulation():
    NA = 1
    t = 0
    A = [1]
    while t < Tmax:

        probs = calcProb(NA)
        NA = nextNA(NA,probs)
        print(t)
        t+=dt
        A.append(NA)

    return A

data = simulation()
ts = np.linspace(0, Tmax, len(data))
plt.plot(ts, data)
plt.xlabel('time (s)')
plt.ylabel('Number of gene A')
plt.show()
