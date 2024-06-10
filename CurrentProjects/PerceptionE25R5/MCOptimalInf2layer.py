from CurrentProjects.PerceptionE25R5.MCBrain2layer import runtime_program, renormalize, nextNs
import numpy as np
from matplotlib import pyplot as plt

Pmnop_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/MCBrain2layer.cl'
MAXTOP=5
MAXBOT = 12
t1 = 2000
Tmax = 2*t1
num_trajectories = 1000
initial = (4,0,0,0) ########Start High A
ULC = 0.8245
LLC = 0.297
##########unbiasedtransitionmatrix
etop1= .0
ebot1 = 0.0                                                                                                                       #kcoop,kcomp,kdu,kud,kx
(halpha, ha,hgamma,hc, kcoop, kcomp,kdu,kud,kx) = (-1 * ULC + etop1 / 2, ULC + -1 * etop1 / 2, -1 * LLC + ebot1 / 2, LLC + -1 * ebot1 / 2, 2.0, 2.43, .8175, .1681, .4359)
unbiasparams = (halpha, ha, halpha - etop1, ha + etop1, hgamma, hc, hgamma - ebot1, hc + ebot1, kcoop, kcomp, kdu, kud, kx)
Punbias = runtime_program(unbiasparams,Pmnop_prog)
Punbiasreshape = Punbias.reshape((MAXTOP, MAXTOP, MAXBOT, MAXBOT, MAXTOP, MAXTOP, MAXBOT, MAXBOT))
Punbiasnormal = renormalize(Punbiasreshape)

##############biasedTransitionmatrix
etop2 = 0
ebot2 = .5                                                                        #kcoop,kcomp,kdu,kud,kx
(halpha, ha,hgamma,hc, kcoop, kcomp,kdu,kud,kx) = (-1 * ULC + etop2 / 2, ULC + -1 * etop2 / 2, -1 * LLC + ebot2 / 2, LLC + -1 * ebot2 / 2, 2.0, 2.43, .8175, .1681, .4359)
biasparams = (halpha, ha, halpha - etop2, ha + etop2, hgamma, hc, hgamma - ebot2, hc + ebot2, kcoop, kcomp, kdu, kud, kx)
Pbias = runtime_program(biasparams,Pmnop_prog)
Pbiasreshape = Pbias.reshape((MAXTOP, MAXTOP, MAXBOT, MAXBOT, MAXTOP, MAXTOP, MAXBOT, MAXBOT))
Pbiasnormal = renormalize(Pbiasreshape)
np.save('bias2layere0_5.npy',Pbiasnormal)

def simulationswitch(Nstart,Punbias,Pbias,Tflip):
    NA = Nstart[0]
    NB = Nstart[1]
    NC = Nstart[2]
    ND = Nstart[3]
    t = 0
    A = [NA]
    B = [NB]
    C = [NC]
    D = [ND]

    while t < Tflip:
        NA, NB,NC,ND = nextNs(Punbias,NA,NB,NC,ND)
        t += 1
        A.append(NA)
        B.append(NB)
        C.append(NC)
        D.append(ND)
    t = 0
    while t<Tflip:
        NA, NB, NC, ND = nextNs(Pbias, NA, NB, NC, ND)
        t += 1
        A.append(NA)
        B.append(NB)
        C.append(NC)
        D.append(ND)

    return A,B,C,D

# Create a list of trajectories for all objects (A, B, C, D)
#unbias = np.load('unbias2layer.npy')
#bias = np.load('bias2layere0_5.npy')
Atrajs =[]
Btrajs =[]
Ctrajs=[]
Dtrajs=[]
for i in range(num_trajectories):
    As,Bs,Cs,Ds =simulationswitch(initial,unbias,bias,t1)
    Atrajs.append(As)
    Btrajs.append(Bs)
    Ctrajs.append(Cs)
    Dtrajs.append(Ds)
    print(i)

def getZ_t(As, Bs):
    Z_t = []
    aminusbs=[]
    for i in range(len(As)):
        aminusb = []
        traja = As[i]
        trajb = Bs[i]
        for t in range(len(traja)):
            dif = traja[t]-trajb[t]
            aminusb.append(dif)
        aminusbs.append(aminusb)

    abstacked = np.stack(aminusbs, axis=0)
    aminusbavg = np.mean(abstacked, axis=0)
    aminusbstd = np.std(abstacked, axis = 0)

    mean = np.mean(aminusbavg)
    std = np.std(aminusbavg)
    print(std)
    for x in range(len(aminusbavg)):
        val = (aminusbavg[x]-mean)/std
        Z_t.append(val)
    return Z_t, aminusbavg , aminusbstd

Zs,average_trajectory, std_deviation_trajectory = (getZ_t(Atrajs,Btrajs))
time_steps = np.arange(0,Tmax+1)

plt.plot(time_steps, average_trajectory, color='black', linewidth=2, label='Average')
plt.fill_between(time_steps, average_trajectory - std_deviation_trajectory, average_trajectory + std_deviation_trajectory, color='r', alpha=0.5, label='±1 Std Dev')
plt.xlabel('Time Step')
plt.ylabel('A-B avg: 1000 samples, epsilon = .5 at t = 2000')
plt.legend()
plt.show()

plt.figure()
ts = np.arange(0,Tmax+1)
plt.plot(ts,Zs,linewidth=1,c='r')
plt.title("A-B Z-score: onset at t = 2000")
plt.show()

