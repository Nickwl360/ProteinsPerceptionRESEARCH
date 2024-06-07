import numpy as np
import sys
from matplotlib import pyplot as plt
from MaxCalAAsim import *
from scipy.optimize import minimize
from numba import jit
from calcPijkMem import*

Pijk_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/MemPijk.cl'

MAX=75
data = np.load('NA_1.npy')
data2=np.load('NA-AAtest.npy')#50k
data3=np.load('NA-AAtest2.npy') #100k
data4=np.load('NA-AAtest30k.npy')#30k
data5=np.load('NA-AAtest500k.npy')#500k
data6=np.load('NA-AAtest100k.npy')#100k

trial1=np.load('test301.npy')
trial2=np.load('test501.npy')
trial3=np.load('test751.npy')
trial4=np.load('test1001.npy')
trial5=np.load('test3001.npy')
trial6=np.load('AA3MillionCorrected.npy')
trial61mil=trial6[:1000000]
trial6300k=trial6[:300000]
trial6100k=trial6[:100000]
trial6200k=trial6[:200000]



memtrial1=np.load('Mem300ktrial1.npy')
memtrial2=np.load('Mem300k1e-3trial1.npy')
memtrial3=np.load('Mem300k5e-3trial1.npy')
memtrial4=np.load('Mem3Milk1e-3trial1.npy')
memtrial5=np.load('Mem3Milk5e-3trial1.npy')
memtrial6=np.load('Mem3Milk2.5e-3trial1.npy')
memtrial7=np.load('Mem10Milk1e-3trial1.npy')

@jit(nopython=True)
def countmatrix(data):
    count = np.zeros((75, 75))
    for i in range(0, len(data) - 1):
        count[data[i]][data[i + 1]] += 1

    return count

@jit(nopython=True)
def Memmatrix(data):
    count = np.zeros((MAX,MAX,MAX))
    for i in range(1, len(data) - 1):
        if i>0:
            count[data[i-1]][data[i]][data[i + 1]] += 1
        # else:
        #     count[0][data[i]][data[i+1]]+=1

    return count
def countmatrix300(data):
    count = np.zeros((75, 75))
    for i in range(0, (len(data) - 1) // 300):
        if i * 300 < len(data) and (i * 300 + 300) < len(data):
            count[data[i * 300]][data[i * 300 + 300]] += 1
    return count
def normalize(pij):
    normfactor=np.zeros(MAX)
    for i in range(0,MAX):
        for j in range(0,MAX):
            normfactor[i] +=pij[i][j]
    for i in range(0,MAX):
        for j in range(0,MAX):
            pij[i][j] /= normfactor[i]
    return pij

@jit(nopython=True)
def Memnormalize(pij):
    normfactor=np.zeros((MAX,MAX))
    for i in range(0,MAX):
        for j in range(0,MAX):
            for k in range(0,MAX):
                normfactor[i][j] +=pij[i][j][k]
    for i in range(0,MAX):
        for j in range(0,MAX):
            for k in range(0,MAX):
                pij[i][j][k] /= normfactor[i][j]
    return pij
def likelyhood(params, count):
    halpha, ha, ka = params

    Pij = calcPij(halpha, ha, ka)  # raise to 300
    pijnorm=normalize(Pij)
    # pij300=np.linalg.matrix_power(Pij,300)
    # count300 = countmatrix300(data)

    #count = countmatrix(data)
    L = 0
    for i in range(0, MAX):
        for j in range(0, MAX):
            if count[i][j] != 0 and pijnorm[i][j] != 0:
                L += count[i][j] * np.log(pijnorm[i][j])
    print(L)
    return -1*L

#@jit(nopython=True)
def Memlikelyhood(params, count):
    halpha, ha, ka,km = params

    ##########cpu
    #Pijktest = calcPijk(halpha, ha, ka,km)  # alreadynormalized

    ##########gpu
    Pijkflat= run_Memprogram(params,Pijk_prog)
    Pijk3=Pijkflat.reshape(MAX,MAX,MAX)
    Pijk=Memnormalize(Pijk3)

    #count = Memmatrix(data)
    L = 0
    for i in range(0, MAX):
        for j in range(0, MAX):
            for k in range(0,MAX):
                if count[i][j][k] != 0: #and Pijk[i][j][k] != 0:
                    L += count[i][j][k] * np.log(Pijk[i][j][k])
    print(L)
    return -1*L
def maximize(count):
    initial =np.array([-0.511, 0.586, 0.0288])
    maxL = minimize(likelyhood, initial, args=(count,), method='Nelder-Mead')
    maxparams = maxL.x
    return maxparams
def memmaximize(count):
    initial =np.array([-0.512, 0.585, 0.0298,.001])
    maxL = minimize(Memlikelyhood, initial, args=(count,), method='Nelder-Mead')
    maxparams = maxL.x
    return maxparams

# compare = [-0.511, 0.586, 0.0288]
# print(likelyhood(compare,data6))
# Pij = calcPij(-0.512, 0.585, 0.0298)
# with np.printoptions(threshold=np.inf):
#     print(Pij)
# Pij10000 = np.linalg.matrix_power(Pij, 10000)
# initial = np.zeros((1,75))
# initial[0]=1
# results = np.matmul(initial,Pij10000)
# total=0
# for i in range(0,75):
#     total+=results[0][i]
# normalresults = [prob/total for prob in results]
# ns = np.linspace(0,75,75)
# plt.plot(ns,normalresults[0])
# plt.show()
#
# #
# print(normalize(calcPij(-0.511, 0.586, 0.0288))[69])
# print(likelyhood((-0.5120435231901056, 0.5831122088601797, 0.02983662952934335),countmatrix(data6)))

# inds = maximize(countmatrix(trial6200k))
# print(inds,likelyhood(inds,countmatrix(trial6200k)))

# #
# inds2 = memmaximize(Memmatrix(memtrial7))
# print(inds2,Memlikelyhood(inds2,Memmatrix(memtrial7)))

#300k.001perterb: -0.51002362  0.58316434  0.0300148   0.00076635  806399.7568564367
#30ktrial1:[-5.19017205e-01  5.91550824e-01  2.93220627e-02  5.63442926e-04] 76747.04623889565
#300ktrial1(frominf): -5.18970540e-01,  5.96263422e-01,  2.91634278e-02,  5.56807752e-04
#300ktrial1(fromtrue):[-5.10976700e-01  5.77991774e-01  3.01059979e-02 -1.27808248e-04] 774791.9330393692

#3MILsanity test
# og=(-.512,.585,.0298,.001)
# #back=(-0.51352667 , 0.55613697 , 0.03163546 , 0.0014804 )
# back=(-0.511674235682794,0.580576875675685,0.030100913608395,0.000798134288955535)
# back10mil= (-0.5111268395146196, 0.5792916417640185, 0.030162653971388204, 0.0007677017182284888)
# print(Memlikelyhood(og,Memmatrix(memtrial7)),Memlikelyhood(back10mil,Memmatrix(memtrial7)))
# #print(memmaximize(Memmatrix(memtrial4)))