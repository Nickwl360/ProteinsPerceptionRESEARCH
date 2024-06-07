import numpy as np
import sys
from matplotlib import pyplot as plt
from MaxCalBrainTest1 import *
from scipy.optimize import minimize
from numba import jit
MAX=26
Pijkl_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/brainpijkl.cl'
dataa= np.loadtxt('NA_sequence.csv',delimiter=",")
datab= np.loadtxt('NB_sequence.csv',delimiter=",")
datalowa=np.loadtxt('MA_sequence.csv',delimiter=",")
datalowb=np.loadtxt('MB_sequence.csv',delimiter=",")
Nalong= np.loadtxt('NA_long_sequence.csv',delimiter=",")
Nblong= np.loadtxt('NB_long_sequence.csv',delimiter=",")
Malong= np.loadtxt('MA_long_sequence.csv',delimiter=",")
Mblong= np.loadtxt('MB_long_sequence.csv',delimiter=",")

# ts=np.linspace(0,len(Nalong),len(Nalong))
# with np.printoptions(threshold=np.inf):
#     print(Nalong)
# plt.plot(ts,Nalong)
# plt.show()


def findepsilon(lowa,lowb):
    epsilon=np.zeros(23)
    for i in range(len(lowb)):
        epsilon[(int(lowb[i]))] += 1
    return epsilon
# count = findepsilon(datalowa,datalowb)
# e = np.linspace(-11,11,len(count))
# maxin = np.where(count == np.max(count))
# print(maxin,max(count))
# plt.plot(e,count)
# plt.show()


def calcm(dataa,datab):
    flips = 0
    for i in range(len(dataa)-1):
        if(abs(dataa[i]-dataa[i+1])>10):
            flips+=1
    mavg = len(dataa)/flips
    return mavg
#print(calcm(dataa,datab))
def spliceconstedata(epsilon,upa,upb,downa,downb):
    newdataa=[]
    newdatab=[]
    for i in range(len(dataa)-1):
        if downa[i]==4 and downb[i] == 3:
            newdataa.append(upa[i])
            newdatab.append(upb[i])
    return newdataa, newdatab
def countmatrix(dataa,datab):
    count = np.zeros((MAX**2, MAX**2),dtype=int)
    for i in range(0, len(datab) - 1):
        count[int(dataa[i]*MAX+datab[i])][int(dataa[i+1]*MAX+datab[i+1])] += 1
    return count
def countmatrixm(dataa, datab,m):
    countm = np.zeros((MAX**2,MAX**2))
    for i in range(0, (len(dataa) - 1) // m):
        if i * m < len(dataa) and (i * m + m) < len(dataa):
            countm[int(dataa[i * m]*25+datab[i * m])][int(dataa[i * m + m]*25+datab[i * m + m])] += 1
    return countm
def likelyhood(params, dataa,datab):

    P = run_program(params,Pijkl_prog)
    Pijklreshape = P.reshape((MAX, MAX, MAX, MAX))
    Pnormal = renormalize(Pijklreshape)
    Pfix = Pnormal.reshape(MAX**2,MAX**2)
    count = countmatrix(dataa,datab)
    m = 226
    countm = countmatrixm(dataa,datab,m)
    Pfixm = np.linalg.matrix_power(Pfix,m)
    L = 0
    for i in range(0, MAX**2):
        for j in range(0, MAX**2):
            if count[i][j] != 0 and Pfix[i][j] != 0:
                L += count[i][j] * np.log(Pfix[i][j])
    print(L)
    return -1*L
def maximize(dataa,datab):
    initial = (0.,0.,0.,0., .1, .1)
    maxL = minimize(likelyhood, initial, args=(dataa,datab,), method='Nelder-Mead')
    maxparams = maxL.x
    return maxparams

a4, b3 = spliceconstedata(1, dataa, datab, datalowa, datalowb)
print(a4[0], b3[0])
ts = np.linspace(0,len(a4),len(a4))
plt.plot(ts,a4,c='blue')
plt.show()
plt.plot(ts, b3, c='red')
plt.show()
print(maximize(a4, b3))


