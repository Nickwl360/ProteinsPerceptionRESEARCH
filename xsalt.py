from SCDcalc import *
from xCalc import *
import math
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

w2 = -1.27
w3 = 0.29
lb = 7.2  # angstrom
l = 3.8
cs = 100  # mM--- mmol/L
k = math.sqrt(8 * math.pi * lb * cs * 6.022E20 / 1e27)

IDPs = getseq('xij_test_seqs.xlsx')
tests = getseq('SCDtests.xlsx')

def A(m, n, x, k):
    return ((.5 * (6 * math.pi / x) ** .5) * (m - n) ** (-1.5) - (
            k * l * math.pi * (2 * (m - n)) ** (-1) * special.erfcx(
        pow((k * l) ** 2 * x * (m - n) / 6, .5))))

def Qsalt(x, seq, k):
    charges = getcharges(seq)
    size = len(charges)
    SUM = 0
    for m in range(2, size + 1):
        nloop = 0
        for n in range(1, m):
            nloop += charges[m - 1] * charges[n - 1] * A(m, n, x, k) * (m - n) ** 2
        SUM += nloop

    return SUM * (1 / size)


def Fsalt(x, seq,k):
    function = 1.5 * (x - np.log(x)) + \
               (3 / (2 * math.pi)) ** (3 / 2) * w2 * (calcOmega(seq)) * (x) ** (-1.5) + \
               w3 * (3 / (2 * math.pi)) ** 3 * calcB(seq) * .5 * (x) ** (-3) + \
               2 * lb * Qsalt(x, seq, k) / (l * math.pi)
    return function


def minXs(seq,k):
    minf = minimize(Fsalt, x0=0.9, args=(seq,k,), method="Nelder-Mead")
    minX = minf.x[0]

    return minX

def getxrray (seq,cray):
    xs=[]
    for i in cray:
        k = math.sqrt(8 * math.pi * lb * i* 6.022e20 / 1e27)
        x = minXs(seq,k)
        xs.append(x)
    return xs

###script
c = np.linspace(0, 1000, 20)
plt.plot(c,getxrray(IDPs[3],c))
w3=.253
w2 = -1
plt.plot(c,getxrray(IDPs[6],c))
w3 =.044
w2 = -.37
plt.plot(c,getxrray(IDPs[7],c))
plt.show()
