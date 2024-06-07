import scipy
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

N= 100000

x1 = 1
testmu = 2.3
testsig = 5

def P(x,mu,sig):
    return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)
def calcarrays():
    x=[x1]
    Px=[P(x1,testmu,testsig)]
    i = 0
    while i <N:
        rand = multivariate_normal.rvs(mean=0,cov=.05)
        xtemp = x[i] + rand
        alpha = min(1,(P(xtemp,testmu,testsig)/P(x[i],testmu,testsig)))
        k= np.random.random()
        if k<=alpha:
            x.append(xtemp)
            Px.append(P(xtemp,testmu,testsig))
        else:
            x.append(x[i])
            Px.append(Px[i])
        i+=1
    return x,Px
def calcPtot(prray):
    Sum=0
    check=0
    for i in range(N):
        Sum += prray[i]
    return Sum

def calcMean(xrray,prray):
    SUM=0
    weight= calcPtot(prray)
    for i in range(N):
        SUM += xrray[i]*prray[i]

    return SUM/weight

def calcvar(xrray,prray):
    SUM=0
    weight= calcPtot(prray)
    N = len(xrray)
    mean = calcMean(xrray,prray)
    for i in range(N):
        SUM+=(xrray[i]-mean)**2
    var = SUM/(N-1)

    # for i in range(N):
    #     SUM += xrray[i]**2*prray[i]/weight
    # secondmom = SUM/N
    #
    # var = secondmom - calcMean(xrray,prray)**2
    return var

xray,pray =calcarrays()

# xs = np.linspace(-6,7,10000)
# ps = P(xs,testmu,testsig)
# plt.plot(xs,ps)
# plt.show()

plt.scatter(xray,pray,s=.5)
plt.show()

print("mean = ",calcMean(xray,pray))
print("variance = " ,calcvar(xray,pray))
print("std = ", np.sqrt(calcvar(xray,pray)))
