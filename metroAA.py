from maxcalAAinference import*
import scipy
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from numba import jit
import csv
stepsize = .02
halpha1=-0.512303955
ha1=.5838539
ka1=.029855265
N = 30000
data= np.load('NA_1.npy')
data2=np.load('NA-AAtest.npy')
data3=np.load('NA-AAtest2.npy')
data4=np.load('NA-AAtest30k.npy')
data6=np.load('NA-AAtest100k.npy')

trial1=np.load('test301.npy')
trial2=np.load('test501.npy')
trial3=np.load('test751.npy')
trial4=np.load('test1001.npy')
trial5=np.load('test3001.npy')
##SEANDATA#########################################################
trial6=np.load('AA3MillionCorrected.npy')
trial61mil=trial6[:1000000]
trial6300k=trial6[:300000]
trial6100k=trial6[:100000]
trial6200k=trial6[:200000]

####################################################################

#SENDINGDATA###############################################################
# trial1list= trial1.tolist()
# trial6list=trial6.tolist()
# filepath= 'AA3MilCorrected.csv'
# with open(filepath, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for value in trial6list:
#         writer.writerow([value])
#######################################################################
def calcRays(count):
    loglmax=-417
    loglmaxbig = - 86553.09938204054 # -0.51202309  0.59223727  0.02967499
    loglmax30k= -76747.81937276587
    loglmax50k = -128420.77798534163
    loglmax75k =-195290.3128081445
    loglmaxhuge = -127956.9103
    loglmax100k = -260469.1306622444 #averages -0.5117244089540235 0.5868207841098471 0.029819665345642392
    loglmax200k=-518171.3062965287
    loglmax100k2=-260469.1306622444
    loglmax300k= - 775118.6379906782  ##first300kfromSEAN
    loglmax1mil=-2587491.3858685405   ##first1MILfromSEAN
    loglmax3mil=-7746027.267264111    ##SEAN
    halpha= [halpha1]
    ha= [ha1]
    ka=[ka1]
    ls = [np.exp(-1 * likelyhood((halpha[0],ha[0],ka[0]),count)-(loglmax200k))]
    accepts=0
    rejects=0
    print(ls[0])
    i = 0
    while i < N:
        rand = multivariate_normal.rvs(mean=[0,0,0], cov=np.diag(np.array([stepsize*.004**2,stepsize*.01**2,stepsize*.0004**2])))
        halphatemp = halpha[i]+rand[0]
        hatemp = ha[i]+rand[1]
        katemp= ka[i]+rand[2]
        lnew= np.exp(-1 *likelyhood((halphatemp,hatemp,katemp),count) - (loglmax200k))
        lold = ls[i]
        print(lnew,lold, accepts, rejects)
        print(lnew/lold,'ln/lo')
        alpha = min(1, (lnew/lold))
        k = np.random.random()
        if k <= alpha: #accept
            accepts+=1
            halpha.append(halphatemp)
            ha.append(hatemp)
            ka.append(katemp)
            ls.append(lnew)
        elif k>alpha: #reject
            rejects+=1
            halpha.append(halpha[i])
            ha.append(ha[i])
            ka.append(ka[i])
            ls.append(ls[i])
        i += 1

    return halpha,ha,ka,ls, accepts,rejects

def calcPtot(prray):
    Sum=0
    for i in range(len(prray)):
        Sum += prray[i]
    return Sum
def calcMean(xrray,prray):
    SUM=0
    weight= calcPtot(prray)
    for i in range(len(prray)):
        SUM += xrray[i]*prray[i]

    return SUM/weight
def calcvar(xrray,prray):
    SUM=0
    N = len(xrray)
    mean = calcMean(xrray,prray)
    for i in range(N):
        SUM+=(xrray[i]-mean)**2
    var = SUM/(N-1)
    return var

#GET STATS
halpha,ha,ka,ls,accepts,rejects = calcRays(countmatrix(trial6200k))
halphabar = calcMean(halpha,ls)
habar = calcMean(ha,ls)
kabar = calcMean(ka,ls)
halphavar = calcvar(halpha,ls)
havar = calcvar(ha,ls)
kavar = calcvar(ka,ls)
print('averages',halphabar,habar,kabar)
print('stds',halphavar**.5,havar**.5,kavar**.5)
print('A / R=', accepts/rejects)

ltotprop=0
probs=np.asarray(ls)
for i in range(len(ls)):
    ltotprop+=ls[i]

normls = probs/ltotprop
counts,bins = np.histogram(ha,bins=50, range=None, density=False, weights=normls)
bar_width=bins[1]-bins[0]

plt.bar(bins[:-1], counts, align='edge', width=bar_width, linewidth=2, edgecolor='blue', color='none')
plt.show()
