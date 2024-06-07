from xCalc import *
import math
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# angstrom
l = 3.8
w2 = 0
w3 = 0.1
omi = 1.8
kb = 1.380649e-23
# These consants include b^3 units
rho = 5
rhot = rho * l ** 3 * 6.022E20 / 1e27
dip = 2.28
dipt = dip / l
IDPs = getseq('xij_test_seqs.xlsx')
tests = getseq('SCDtests.xlsx')

def fminus(seq):
    N = len(seq)
    qs = getcharges(seq)
    count = sum(1 for num in qs if num == -1)
    return count / N
def fplus(seq):
    N = len(seq)
    qs = getcharges(seq)
    count = sum(1 for num in qs if num == 1)
    return count / N
def calcK(a1, a2, seq,t,cs):
    lb = 2083.657758 / (t+273.15)
    lbt = lb / l
    cst = cs * l ** 3 * 6.022E20 / 1e27  # mM--- mmol/L
    return (4 * math.pi * lbt * ((fplus(seq) * a1 + fminus(seq) * a2) * rhot + 2 * cst)) ** .5
def F1(a1, a2, seq,t):
    return len(seq)*kb*(t+273.15) * (fplus(seq) * (a1 * np.log(a1) + (1 - a1) * np.log(1 - a1)) + fminus(seq) * (
            a2 * np.log(a2) + (1 - a2) * np.log(1 - a2)))
def F2(a1, a2, seq,t,cs):
    cst = cs * l ** 3 * 6.022E20 / 1e27  # mM--- mmol/L
    return len(seq) *kb*(t+273.15) *((fplus(seq) * a1 + cst / rhot) * np.log(fplus(seq) * a1 * rhot + cst) + (
            fminus(seq) * a2 + cst / rhot) * np.log(fminus(seq) * a2 * rhot + cst) - (
                               fplus(seq) * a1 + fminus(seq) * a2 + 2 * cst / rhot))
def F3(a1, a2, seq,t,cs):
    lb = 2083.657758 / (t+273.15)
    lbt = lb / l
    cst = cs * l ** 3 * 6.022E20 / 1e27  # mM--- mmol/L
    return -2 * len(seq) *kb*(t+273.15) * (math.sqrt(math.pi) * lbt ** 1.5 * (
            (fplus(seq) * a1 + fminus(seq) * a2) * rhot + 2 * cst) ** (
               1.5) / (3 * rhot))
def F4(a1, a2, seq,t):
    lb = 2083.657758 / (t+273.15)
    lbt = lb / l
    return -1 * len(seq) *kb*(t+273.15) * (fplus(seq) * (1 - a1) + fminus(seq) * (1 - a2)) * lbt * (omi + .5) / dipt
def F5(x, a1, a2, seq,t,cs):
    lb = 2083.657758 / (t+273.15)
    lbt = lb / l
    function = kb*(t+273.15) *(1.5 * (x - np.log(x)) + w3 * (3 / (2 * math.pi * x)) ** 3 * calcB(seq) * .5 + \
               2 * lbt * Qcon(x, a1, a2, seq,t,cs) * (3 / (2 * math.pi * x)) ** .5 + (3 / (2 * math.pi * x)) ** (
                   1.5) * Omega(a1, a2, seq,t,cs))
    return function
def qmqn(a1, a2, index, seq):
    sign = getcharges(seq)[index - 1]
    if sign == 1:
        return a1
    elif sign == -1:
        return -1 * a2
    elif sign == 0:
        return 0
def cmn(a1, a2, index, seq):
    sign = getcharges(seq)[index - 1]
    if sign == 1:
        return a1
    elif sign == -1:
        return a2
    elif sign == 0:
        return 0
def dmn(a1, a2, index, seq):
    sign = getcharges(seq)[index - 1]
    if sign == 1:
        return 1-a1
    elif sign == -1:
        return 1-a2
    elif sign == 0:
        return 0
def Qcon(x, a1, a2, seq,t,cs):
    N = len(seq)
    qs = getcharges(seq)
    SUM = 0
    for m in range(2, N + 1):
        nloop = 0
        for n in range(1, m):
            nloop += qmqn(a1, a2, m, seq) * qmqn(a1, a2, n, seq) * A(
                calcK(a1, a2, seq,t,cs) ** 2 * x * (m - n) / 6) * (m - n) ** .5
        SUM += nloop

    return SUM * (1 / N)
def A(z):
    return 1 - np.sqrt(math.pi * z) * special.erfcx(np.sqrt(z))
def Omega(a1, a2, seq,t,cs):
    Onone = w2 * calcOmega(seq)
    Ocd = ocdCalc(a1, a2, seq,t,cs)
    Odd = oddCalc(a1, a2, seq,t,cs)

    return Onone + Ocd + Odd
def ocdCalc(a1, a2, seq,t,cs):
    size = len(seq)
    SUM = 0
    lb = 2083.657758 / (t+273.15)
    lbt = lb / l
    wcd = - math.pi * omi ** 2 * lbt ** 2 * dipt ** 2 * np.exp(-2 * calcK(a1, a2, seq,t,cs)) * (2 + calcK(a1, a2, seq,t,cs)) / 3
    for m in range(2, size + 1):
        nloop = 0
        for n in range(1, m):
            nloop += (m - n) ** (-(1 / 2)) * (
                    cmn(a1, a2, m, seq) * (dmn(a1, a2, n, seq)) + cmn(a1, a2, n, seq) * (
                    dmn(a1, a2, m, seq)))
        SUM += nloop
    return SUM * (wcd / size)
def oddCalc(a1, a2, seq,t,cs):
    size = len(seq)
    SUM = 0
    lb = 2083.657758 / (t+273.15)
    lbt = lb / l
    wdd = - math.pi * omi ** 2 * lbt ** 2 * dipt ** 4 * \
          np.exp(-2 * calcK(a1, a2, seq,t,cs)) * (
                  4 + 8 * calcK(a1, a2, seq,t,cs) + 4 * calcK(a1, a2, seq,t,cs) ** 2 + calcK(a1, a2, seq,t,cs) ** 3) / 9
    for m in range(2, size + 1):
        nloop = 0
        for n in range(1, m):
            nloop += (m - n) ** (-(1 / 2)) * (dmn(a1, a2, m, seq)) * (dmn(a1, a2, n, seq))
        SUM += nloop
    return SUM * (wdd / size)

