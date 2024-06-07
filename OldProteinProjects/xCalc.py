from SCDcalc import *
from scipy.optimize import minimize
import math

# constants
w2 = -0.4
w3 = 0.1
lb = 7.2
l = 3.8

IDPs = getseq('xij_test_seqs.xlsx')


def calcOmega(seq):
    size = len(seq)
    SUM = 0
    nloop = 0
    for m in range(2, size + 1):

        nloop = 0
        for n in range(1, m):
            nloop += (m - n) ** (-(1 / 2))
        SUM += nloop
    # include last sum

    return SUM * (1 / size)


def calcB(seq):
    size = len(seq)
    SUM = 0
    mloop = 0
    nloop = 0

    for p in range(3, size + 1):
        mloop = 0
        for m in range(2, p):
            nloop = 0
            for n in range(1, m):
                nloop += (p - n) / (((p - m) * (m - n)) ** (3 / 2))
            mloop += nloop
        SUM += mloop
    return SUM * (1 / size)


def F(x, seq):
    function = 1.5 * (x - math.log(x, math.e)) + \
               (3 / (2 * math.pi)) ** (3 / 2) * w2 * (calcOmega(seq)) * (x) ** (-1.5) + \
               w3 * (3 / (2 * math.pi)) ** 3 * calcB(seq) * .5 * (x) ** (-3) + \
               lb * calcQ(seq) * math.sqrt(6 / math.pi) / (l * x ** (.5))
    return function


def minX(seq):
    minf = minimize(F, x0=0.5, args=(seq,))
    minX = minf.x[0]

    return minX

# for i in range(len(IDPs)):
# print(minX(IDPs[0]))
