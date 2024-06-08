from SCDcalc import *
from xCalc import *
import math
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from CondensationFunctions import *


##a1 = a+, a2 = a-
def Fcon(params, seq):
    x, a1, a2 = params
    function = F1(a1, a2, seq) + F2(a1, a2, seq) + F3(a1, a2, seq) + F4(a1, a2, seq) + F5(x, a1, a2, seq)
    return function

# def minXs(seq,k):
# minf = minimize(Fcon, x0=0.9, args=(seq,k,), method="Nelder-Mead")
# minX = minf.x[0]

# return minX
seq = getseq("SCDtests.xlsx")
seq2 = getseq("xij_test_seqs.xlsx")
initial_guess = (1.2, .97, .9)
#bounds = [(0, 10), (0, 10), (0, 11)]
#,bounds=bounds

result = minimize(Fcon, initial_guess, args=(seq[0],), method="Nelder-Mead",)
optimized_values = result.x
xmin,a1min,a2min = optimized_values
print(f"x: {xmin}, a1: {a1min}, a2: {a2min}")




