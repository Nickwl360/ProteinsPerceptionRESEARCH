import numpy as np
import matplotlib.pyplot as plt
from MCBrain2layer import runtime_program
from MCBrain2layer import renormalize
from MCBrain2layer import nextNs
import scipy.io

TMAX = 300000
# samples = 300000
MAXTOP = 5
MAXBOT = 12
# Pmnop_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/MCBrain2layer.cl'

jochdata = scipy.io.loadmat('StochasticRecurrentSymmetric (1).mat')
print(jochdata.keys())
Rkij = jochdata['R_kij']
Ekij = jochdata['E_kij']
set = 0
Upleft=Rkij[0,:,set]
Upright=Rkij[1,:,set]
Botleft=Ekij[0,:,set]
Botright=Ekij[1,:,set]
dataa = np.round(Upleft[:]*4)
datab = np.round(Upright[:]*4)
datac = np.round(Botleft[:]*11)
datad = np.round(Botright[:]*11)

# controla = np.load('Atest3.npy')
# controlb = np.load('Btest3.npy')
# controlc = np.load('Ctest3.npy')
# controld = np.load('Dtest3.npy')

JochFwrAset15 = np.load('JochenFwrboundset15A.npy')
JochFwrBset15 = np.load('JochenFwrboundset15B.npy')
JochFwrCset15 = np.load('JochenFwrboundset15C.npy')
JochFwrDset15 = np.load('JochenFwrboundset15D.npy')
#
# JochFwrAset0 = np.load('JochenFwrset0A.npy')
# JochFwrBset0 = np.load('JochenFwrset0B.npy')
# JochFwrCset0 = np.load('JochenFwrset0C.npy')
# JochFwrDset0 = np.load('JochenFwrset0D.npy')

###FULLRESET
# initial=(0,4,0,0) #Tmax = 3000
# epsilon1= .0
# etop2 = .45                                                                           #kcoop,kcomp,kdu,kud,kx
# (halpha, ha,hgamma,hc, kcoop, kcomp,kdu,kud,kx) = ( epsilon1/2,-1*epsilon1/2,etop2/2, -1*etop2/2,1,2,.2,.05,.5)
# params = (halpha, ha, halpha - epsilon1, ha + epsilon1,hgamma,hc,hgamma-etop2,hc +etop2, kcoop, kcomp,kdu,kud,kx)
# initial=(0,4,0,0)  #A,B,C,D
# ULC = 0.8245
# LLC = 0.297
# epsilon1= .0
# epsilon2 = 1                                                                                                                       #kcoop,kcomp,kdu,kud,kx
# # (halpha, ha,hgamma,hc, kcoop, kcomp,kdu,kud,kx) = ( -1 * ULC + epsilon1/2,ULC + -1*epsilon1/2, -1*LLC +etop2/2,LLC + -1*etop2/2,0,0,0,.5,0)
# (halpha, ha,hgamma,hc, kcoop, kcomp,kdu,kud,kx) = ( -1 * ULC + epsilon1/2,ULC + -1*epsilon1/2, -1*LLC +epsilon2/2,LLC + -1*epsilon2/2,2.0,2.43,.8175,.1681,.4359)
# #(halpha, ha,hgamma,hc, kcoop, kcomp,kdu,kud,kx) = ( .0,.0,.0, .0 ,.0,.0,.0,.0,.0)
# params = (halpha, ha, halpha - epsilon1, ha + epsilon1,hgamma,hc,hgamma-epsilon2,hc +epsilon2, kcoop, kcomp,kdu,kud,kx)
# #

# Pmnop = runtime_program(params,Pmnop_prog)
# pmnopreshape = Pmnop.reshape((MAXTOP, MAXTOP, MAXBOT, MAXBOT, MAXTOP, MAXTOP, MAXBOT, MAXBOT))
# pmnopnormal = renormalize(pmnopreshape)
# np.save('Gamma2layer1.npy',pmnopnormal)
# PGammaDist = np.load('Gamma2layer1.npy')


def simtilflip(Nstart,Tmax,P):
    NA = Nstart[0]
    NB = Nstart[1]
    NC = Nstart[2]
    ND = Nstart[3]
    t = 0
    check =0
    flip = 0
    Pijklnormal=P
    while t < Tmax:
        NA, NB,NC,ND = nextNs(Pijklnormal,NA,NB,NC,ND)
        if NB<NA:
            check +=1
            if check == 10:
                flip = t - 10
                return flip
        t += 1
def sampletimes(initial,samples,P):
    fliptime=0
    i = 0
    toggletimes=[]
    while i < samples:
        toggletimes.append(simtilflip(initial,TMAX,P))
        i+=1
        print(i)
    return toggletimes
def countFliptimes(dataA, dataB):
    fliptimes=[]
    time=0
    if dataA[2]>dataB[2]:
        A,B=1,0
    else: A,B = 0,1
    for i in range(3,len(dataA)):
        if A == 1:
            if dataA[i]>=dataB[i]:
                time+=1
            else:
                B,A=1,0
                fliptimes.append(time)
                time=0
        if B ==1:
            if dataB[i]>=dataA[i]:
                time+=1
            else:
                A,B = 1,0
                fliptimes.append(time)
                time=0
    return fliptimes
def getActivityDist(dataA,dataB,dataC,dataD):
    ABDist=np.zeros(MAXTOP)
    CDDist=np.zeros(MAXBOT)
    for i in range(len(dataA)):
        a = int(dataA[i])
        c = int(dataC[i])
        ABDist[a]+=1
        CDDist[c]+=1
    ABDist/=np.sum(ABDist)
    CDDist/=np.sum(CDDist)
    return ABDist, CDDist

# ##V1###############
nbins=60
#toggletimes =sampletimes(initial,samples,PGammaDist)
AdistJoch,CdistJoch= getActivityDist(dataa,datab,datac,datad)
# AdistControl,CdistControl=getActivityDist(controla,controlb,controlc,controld)
AdistJochFwr,CdistJochFwr=getActivityDist(JochFwrAset15,JochFwrBset15,JochFwrCset15,JochFwrDset15)

toggletimes = countFliptimes(dataa,datab)
# toggletimes2 = countFliptimes(controla,controlb)
# toggletimes3 = countFliptimes(JochFwrAset15,JochFwrBset15)
toggletimes5 = countFliptimes(JochFwrAset15,JochFwrBset15)
print(toggletimes)
np.save('MC2layerflipsJochenset16v2',toggletimes5)
#np.save('MC2layerflips1milcontrol',toggletimes2)
#np.save('MC2layerflips1milJochFwrSet16',toggletimes3)
np.save('MC2layerflips1_5milJochFwrSet0',toggletimes5)


#np.save('MC2layerflips1.npy',toggletimes)
#times = np.load("MC2layerflips1.npy")
# R = np.ptp(times)
# nbinswidth = R*(samples**(1/3)/(3.49*std))
# nbinsf= np.sqrt(samples)
# nbins = int(round(nbinsf))
plt.hist(toggletimes,bins=nbins)
plt.title('jochens flip distribution')
plt.figure()
# plt.hist(toggletimes2,bins=nbins)
# plt.title('my origninal parameters')
# plt.figure()
# plt.hist(toggletimes3,bins=nbins)
# plt.title('Fwrd Jochen infered parameters')
# plt.show()
plt.hist(toggletimes5,bins=nbins)
plt.title('Fwrd Jochen infered parameters')
plt.show()
numsT = np.arange(MAXTOP)
numsB= np.arange(MAXBOT)
plt.figure()
plt.bar(numsT,AdistJoch)
plt.title('A Activity Jochen Distribution')
plt.figure()
plt.bar(numsB,CdistJoch)
plt.title('C Activity Jochen Distribution')
plt.figure()

plt.bar(numsT,AdistJochFwr)
plt.title('A Activity Jochen Fwr infered Distribution')
plt.figure()
plt.bar(numsB,CdistJochFwr)
plt.title('C Activity Jochen Fwr infered Distribution')

plt.show()