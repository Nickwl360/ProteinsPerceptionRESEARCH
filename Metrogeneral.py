import numpy as np
from scipy.stats import multivariate_normal
from maxcalAAinference import*
from MCBrain2LayerInference import *


#memorystuff
# trial1=np.load('AA3MillionCorrected.npy')
# memtrial2=np.load('Mem300k1e-3trial1.npy')
# memtrial3=np.load('Mem300k5e-3trial1.npy')
# memtrial4=np.load('Mem3Milk1e-3trial1.npy')
# memtrial5=np.load('Mem3Milk5e-3trial1.npy')
# memtrial6=np.load('Mem3Milk2.5e-3trial1.npy')
# memtrial7=np.load('Mem10Milk1e-3trial1.npy')

#MCbrain2layer
#########1mil
#brain1a, brain1b,brain1c,brain1d = np.load('Atest3.npy'),np.load('Btest3.npy'),np.load('Ctest3.npy'),np.load('Dtest3.npy')

#########10mil
brain1a, brain1b,brain1c,brain1d = np.load('Atest10mil.npy'),np.load('Btest10mil.npy'),np.load('Ctest10mil.npy'),np.load('Dtest10mil.npy')
#print(len(brain1a))

def countFlips(seqA,seqB):
    # flips = 0
    # for i in range(len(seqA)-1):
    #     if seqA[i] > seqB[i] and seqA[i+1] < seqB[i+1]:
    #         flips +=1
    #     if seqB[i] > seqA[i] and seqB[i+1] < seqA[i+1]:
    #         flips +=1
    # return flips, len(seqA)
    time=0
    flips = 0
    counter =0
    for i in range(len(seqA)-1):
        if seqA[i]>seqB[i] and seqA[i+1]>seqB[i+1]:
            counter +=1
        elif seqB[i]>seqA[i] and seqB[i+1]>seqA[i+1]:
            counter +=1
        else:
            time+=counter
            counter = 0
            flips+=1
    return time/flips
#f, l= countFlips(brain1a,brain1b)
f=countFlips(brain1a,brain1b)
print(f,len(brain1a)/f)

def calcrays(likelyhood,initial,count,lmax,N=30000):
    #brain1mil  stepsize=.41
    stepsize=.03
    length=1_500_001

    #need lagrange multipliers array, and likelyhood array
    multipliers = np.zeros((len(initial),N))  #array of arrays
    for i in range(len(initial)):
        multipliers[i][0]=initial[i]
    likelyhoods=np.zeros(N)
    likelyhoods[0]=np.exp((-1*likelyhood((initial),count)-lmax)*length)
    accepts=0
    rejects=0

    i=0

    #braindiag7
    diag=np.array([stepsize*.004**2,stepsize*.003**2,stepsize*.004**2,stepsize*.004**2,stepsize*.004**2,stepsize*.001**2,stepsize*.004**2]) #FIXTHISFORBRAIN

    #diagsean+mem
    #diag=np.array([stepsize*.004**2,stepsize*.01**2,stepsize*.0004**2]) #SEAN3mil
    #diag=np.array([stepsize*.004**2,stepsize*.01**2,stepsize*.0004**2,stepsize*.0002**2]) #memorytake2
    #diag=np.array([stepsize*.004**2,stepsize*.01**2,stepsize*.0004**2,stepsize*.0001**2]) #memorytake3 cov[4]=.003/.001(take4)
    #diag=np.array([stepsize*.004**2,stepsize*.01**2,stepsize*.0004**2,stepsize*.00001**2]) #memory3Mil

    while i<N-1:
        rand=multivariate_normal.rvs(mean=np.zeros(len(initial)),cov= np.diag(diag))
        temp = np.zeros(len(initial))
        for j in range(len(initial)):
            temp[j]=multipliers[j][i]+rand[j]
        length=10_000_001
        lold = likelyhoods[i]
        lnew=np.exp((-1*likelyhood((temp),count)-lmax)*length)

        print('new: ',lnew, 'old: ',lold)
        print('LNew/Lold: ',lnew/lold)
        alpha = min(1, (lnew / lold))
        k = np.random.random()
        print(k,alpha)
        if k <= alpha:  # accept
            accepts += 1
            for j in range(len(initial)):
                multipliers[j][i+1] = temp[j]
                likelyhoods[i+1]=lnew
        elif k > alpha:  # reject
            rejects += 1
            for j in range(len(initial)):
                multipliers[j][i+1]=multipliers[j][i]
                likelyhoods[i+1]=lold
        i += 1
        print('Accepts,Rejects: ',accepts,rejects)
        print('                             ')
    return multipliers, likelyhoods , accepts,rejects
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
def printStats(hs,ls):
    avgs =[]
    for i in range(len(hs)):
        avgs.append(calcMean(hs[i],ls))
    print('avg multipliers: ', avgs)
    stds=[]
    for i in range(len(hs)):
        stds.append(calcvar(hs[i], ls)**.5)
    print('stds: ', stds)

    return
def displayhist(ls,h):
    ltotprop = 0
    probs = np.asarray(ls)
    for i in range(len(ls)):
        ltotprop += ls[i]

    normls = probs / ltotprop
    counts, bins = np.histogram(h, bins=50, range=None, density=False, weights=normls)
    bar_width = bins[1] - bins[0]

    plt.bar(bins[:-1], counts, align='edge', width=bar_width, linewidth=2, edgecolor='blue', color='none')
    plt.show()
    return
#memtrials3mil
# initial=(-0.511, 0.586, 0.0288)
# lmax =-7746027.267264111
# # hs,ls,accepts,rejects = calcrays(likelyhood,initial,countmatrix(trial1),lmax)
# # printStats(hs,ls)
# # print(calcMean(hs[0],ls),calcMean(hs[1],ls),calcMean(hs[2],ls))
# # print(calcvar(hs[0],ls)**.5,calcvar(hs[1],ls)**.5,calcvar(hs[2],ls)**.5)
#memorytest .001 perterbation
# initial=(-0.511, 0.586, 0.0288,.0011)
# initial2=(-0.511, 0.586, 0.0288,.0051)
# initial3mil=(-0.5120996,   0.579035152 , 0.02991496 , 0.0008099)
# initial3mil2=(-0.52578454,  0.39416949 , 0.0417734 ,- 0.00135243)
# initial3mil25=(-0.51352666 , 0.55613696 , 0.03163545 ,  0.00148041)
# initial10mil=(-0.51109306 , 0.57904936 , 0.03017862,  0.00075729)

# lmax =-806399.7568564367
# lmax2 =-804805.5129090043
# lmax3= -8057602.397390642
# lmax4=-8046527.2613321785
# lmax5=-8127019.100673093
# lmax6=-26879496.673876252
#-0.51352667  0.55613697  0.03163546  0.0014804 ] 8127019.100673093andpijk!=0
#[-0.51357865  0.55581275  0.03165086  0.00147715] 8127132.362148795 :
#[-0.51109307  0.57904935  0.03017861  0.00075728] 10mil
# hs,ls,accepts,rejects = calcrays(Memlikelyhood,initial10mil,Memmatrix(memtrial7),lmax6)
# printStats(hs,ls)


#COMPARESTRATS
#3MIL
# og=(-.512,.585,.0298,.001)
# back=(-0.51352667 , 0.55613697 , 0.03163546 , 0.0014804 )
# print(Memlikelyhood(og,Memmatrix(memtrial4)),Memlikelyhood(back))

#np.save("stats",printStats(hs,ls))
#displayhist(ls,hs[3])
#brainstuff
#1mil
#initial=(8.54e-01, 3.035e-01,4.37e-04,1.97e+00,2.40e+00,8.10e-01,1.66e-01,4.27e-01)
#lmax =-1.6326069734605666
#10mil
lmax= -1.6337383676975317
initial=(8.54e-01, 3.035e-01,1.97e+00,2.40e+00,8.10e-01,1.66e-01,4.27e-01)

hs,ls,accepts,rejects = calcrays(brainlikelyhood,initial,countbrain(brain1a, brain1b,brain1c,brain1d),lmax)
printStats(hs,ls)

#1MILBRAIN step = .4 A,R = 993,4006
#avg multipliers:  [0.8777360969969095, 0.3035849343851974, 0.00011360828220658071, 1.9771785842273901, 2.4064228446820595, 0.8116864589583125, 0.16680438260290162, 0.4260705059846524]
#stds:  [0.025503050753284493, 0.001939875386393769, 0.0006358351474561973, 0.009277591611430228, 0.007392413507666539, 0.0049704934662050775, 0.0006752958058968336, 0.005114857842344811]

