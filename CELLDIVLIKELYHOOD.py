from MCCELLDIV import *
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
import sys
from scipy.optimize import minimize

m=6
MAX = 89
# multipliers=(-0.583 , .266,.0376,-4.66,-.04)
celldatapath = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/celldivdata/'
nodivcount = np.load('trajectory_counts_nodiv-3.npy')
divcount = np.load('trajectory_counts_div-3.npy')
nicknodiv2 =np.load('ndivv2.npy')
nickdiv2 = np.load('divv2.npy')

fluri = np.load('OldMCProjects/trans_nodiv_i.npy')
flurj = np.load('OldMCProjects/trans_nodiv_j.npy')
flurdi = np.load('OldMCProjects/trans_div_i.npy')
flurdk = np.load('OldMCProjects/trans_div_k.npy')
flurdl = np.load('OldMCProjects/trans_div_l (1).npy')


# def Likelyhood(multipliers,divcount,nodivcount,m):
#     L = 0
#     pdiv,pnodiv,ppost = calceitherPij(multipliers)
#     pnodiv = np.transpose(pnodiv)
#     pdiv = np.transpose(pdiv)
#     ppost = np.transpose(ppost)
#
#     ###NODIVCONTRIBUTION
#     pnodivm= np.linalg.matrix_power(pnodiv,m)
#     for i in range(MAX+1):
#         for j in range(MAX+1):
#             if nodivcount[j][i]!=0:
#                 L += np.log(pnodivm[j][i])*nodivcount[j][i]
#     ###DIVCONTRIBUTION
#     pikl = np.zeros((MAX+1,MAX+1,MAX+1),np.float64)
#     for n in range(1, m + 1):
#         pijpre = np.linalg.matrix_power(pnodiv, n - 1)
#         pjpost = np.linalg.matrix_power(pnodiv, m - n)
#         # secondterm = np.matmul(pjpost,pdiv)
#         # thirdterm= np.matmul(pjpost,ppost)
#         secondterm = pjpost @ pdiv
#         thirdterm = pjpost @ ppost
#
#         for i in range(0,MAX+1):
#             for k in range(0,MAX+1):
#                 for l in range(0,MAX+1):
#                     for j in range(0, MAX + 1):
#                         pikl[i][k][l] += (pijpre[j][i] * secondterm[k][j] * thirdterm[l][j])
#
#     for a in range(MAX + 1):
#         for b in range(MAX+1):
#             for c in range(MAX+1):
#                 if divcount[b][c][a]!=0:
#                     L += np.log(pikl[a][b][c])*divcount[b][c][a]
#
#     print(L)
#     return -1*L

F_f0 =100
F_sig = 30
@jit(nopython=True)
def Pfn(f,a,b,n):
    if n>0:
        pfn = 1. / (np.sqrt(2. * np.pi*b**2*n)) * np.exp(-np.power((f - n*a),2)/(2*n*b**2))
    else: pfn = 0
    return pfn

@jit(nopython=True)
def phi(pneq,f):

    phi = np.zeros(MAX+1)
    if f !=0:
        for i in range(MAX+1):
            phi[i]= pneq[i]*Pfn(f,F_f0,F_sig,i)

        phi = phi / sum(phi)
        phi[0]=0

    else: phi[0]=1

    return phi
@jit(nopython=True)
def Pnequ(p):
    plarge = np.linalg.matrix_power(p,10000)
    i = np.zeros(MAX+1)
    i[0]=1
    pequ = i @ plarge #np.matmul(i,plarge)
    return pequ
@jit(nopython=True)
def calcpikl(pnodiv,pdiv,ppost):
    pikl = np.zeros((MAX+1, MAX+1, MAX+1), np.float64)

    for n in range(1, m + 1):
        pijpre = np.linalg.matrix_power(pnodiv, n - 1)
        pjpost = np.linalg.matrix_power(pnodiv, m - n)
        # pijpre = pijpre.transpose()
        # # pjpost = pjpost.transpose()
        # pdiv = pdiv.transpose()
        # ppost = ppost.transpose()

        secondterm = pjpost @ pdiv
        thirdterm = pjpost @ ppost

        pijpre=pijpre.transpose()
        secondterm = secondterm.transpose()
        thirdterm=thirdterm.transpose()

        for i in range(0, MAX+1):
            for k in range(0, MAX+1):
                for l in range(0, MAX+1):
                    for j in range(0, MAX+1):
                        # pikl[i][k][l] += (pijpre[j][i] * secondterm[k][j] * thirdterm[l][j])
                        pikl[l][k][i] += (pijpre[j][i] * secondterm[k][j] * thirdterm[l][j])

    return pikl


@jit(nopython=True)
def FlurLikelyhood(multipliers,fluri,flurj,flurdi,flurdk,flurdl,m):
    L = 0
    pdiv,pnodiv,ppost = calceitherPij(multipliers)
    # pnodiv = np.transpose(pnodiv)
    # pdiv = np.transpose(pdiv)
    # ppost = np.transpose(ppost)
    Peq = Pnequ(pdiv + pnodiv)
    phiis = []
    phijs = []
    phidis=[]
    phidks=[]
    phidls=[]

    for i in range(len(fluri)):
        phii = phi(Peq,fluri[i])
        phiis.append(phii)
        phij = phi(Peq,flurj[i])
        phijs.append(phij)
    for i in range(len(flurdi)):
        phidi = phi(Peq, flurdi[i])
        phidis.append(phidi)
        phidk = phi(Peq, flurdk[i])
        phidks.append(phidk)
        phidl = phi(Peq, flurdl[i])
        phidls.append(phidl)
    ###NODIVCONTRIBUTION
    pnodivm= np.linalg.matrix_power(pnodiv,m)
    pnodivm = pnodivm.transpose()
    for i in range(len(phiis)):
        phi1 = phiis[i]
        phi2 = phijs[i]
        sum = phi2 @ pnodivm @ phi1
        L += np.log(sum)
    print(L)
    ###DIVCONTRIBUTION
    pikl = calcpikl(pnodiv,pdiv,ppost)
    # pikl = pikl.transpose()
    for a in range(len(phidis)):
        phii = phidis[a]
        phik = phidks[a]
        phil = phidls[a]
        temp=0
        for i in range(MAX+1):
            for k in range(MAX+1):
                for l in range(MAX+1):
                    temp +=pikl[l][k][i] * phii[i] * phik[k] * phil[l]
                    # temp +=pikl[i][k][l] * phii[i] * phik[k] * phil[l]

        L += np.log(temp)
    return -1*L

def maximize(fluri,flurj,flurdi,flurdk,flurdl,m):
    initial = (-0.582, .265, .0372, -4.62, -.035)
    maxL = minimize(FlurLikelyhood, initial, args=(fluri,flurj,flurdi,flurdk,flurdl,m,), method='Nelder-Mead')
    maxparams = maxL.x
    return maxparams


maxl = (-0.5845411 , 0.12693621, 0.04277835, -4.66577823, -0.03983156)
print(FlurLikelyhood(maxl,fluri,flurj,flurdi,flurdk,flurdl,m))
# maxparams = maximize(maxl,fluri,flurj,flurdi,flurdk,flurdl,m)
# print('maxparams = ', maxparams)   # [-1.87778625e-01  1.16157704e-01  2.25217701e-02 -3.32023980e+01 -1.73974753e-01]
# multiplies = (-0.5845411 , 0.12693621, 0.04277835, -4.66577823, -0.03983156)
# print(FlurLikelyhood(multiplies,fluri,flurj,flurdi,flurdk,flurdl,m))