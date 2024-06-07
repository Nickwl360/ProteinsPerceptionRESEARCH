
from xijFunctions import *
from scipy.optimize import minimize
import math

IDPs = getseq("xij_test_seqs.xlsx")

#constants
w2=-0.4
w3=0.1
lb=7.2
l=3.8

#ij input
i=30
j=5

def Fij(x,seq,i,j):
    function = 1.5*(x-math.log(x,math.e)) + \
              (3/(2*math.pi))**(3/2)*w2* (calcSHij(seq,i,j)) * (x)**(-1.5) + \
              w3*(3/(2*math.pi))**3 * calcTij(seq, i, j)*.5*(x)**(-3) + \
              lb*calcSDij(seq,i,j)*math.sqrt(6/math.pi)/(l*x**(.5))
    return function

def  minXij(seq,i,j):
    minf = minimize(Fij, x0=.66, args=(seq,i,j,), method= 'Nelder-Mead')
    minX = minf.x[0]

    return minX


print(minXij(IDPs[0],i,j))
print(minXij(IDPs[3],i,j))
print(minXij(IDPs[4],i,j))