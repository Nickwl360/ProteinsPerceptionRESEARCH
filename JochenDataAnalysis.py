import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def countUps(upL,upR):
    l = 0
    r = 0
    for i in range(len(upL)):
        if upL[i]>upR[i]:
            l+=1
        if upR[i]>upL[i]:
            r+=1
    return l, r
set = 0
NE = 25
NR = 5
rang=63000
low = 13000
jochdata = scipy.io.loadmat('StochasticRecurrentSymmetric (1).mat')
#jochdatafix = scipy.io.loadmat('StochasticRecurrentSymmetricNE11NR4.mat')
jochdatafix = scipy.io.loadmat('StochasticRecurrentSymmetricNE25NR5.mat')
print(jochdatafix.keys())

hspredicted = scipy.io.loadmat('BaselineMultipliersDT001.mat')
#IS = hspredicted['I_n']
#print(IS)
#heplus = hspredicted['hR_plus_n']
Rkij = jochdatafix['R_kij']
Ekij = jochdatafix['E_kij']
times = jochdatafix['dt']
cs = jochdatafix['C_0']
print(cs)
print(times)
print(Rkij.shape)
Upleft=Rkij[0,:,set]
Upright=Rkij[1,:,set]
Botleft=Ekij[0,:,set]
Botright=Ekij[1,:,set]

UL = Upleft[low:rang]/NR
UR = Upright[low:rang]/NR
BL = Botleft[low:rang]/NE
BR = Botright[low:rang]/NE

#####OUTDATED OLD TRAJECTS
# Rkij = jochdata['R_kij']
# Ekij = jochdata['E_kij']
# times = jochdata['dt']
# print(times)
# print(Rkij.shape)
# Upleft=Rkij[0,:,set]
# Upright=Rkij[1,:,set]
# Botleft=Ekij[0,:,set]
# Botright=Ekij[1,:,set]
# UL = np.round(Upleft[low:rang]*4)
# UR = np.round(Upright[low:rang]*4)
# BL = np.round(Botleft[low:rang]*11)
# BR = np.round(Botright[low:rang]*11)
# print(countUps(UL,UR))
t = np.arange(0,len(UL))
plt.plot(t,UR,c='r')
plt.plot(t,UL,c='b')
plt.title('toplayer')
plt.figure()
plt.plot(t,BR,c='r')
plt.plot(t,BL,c='b')
plt.title('botlayer')
plt.show()
