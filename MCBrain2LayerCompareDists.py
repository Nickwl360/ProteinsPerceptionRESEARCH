import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import os
#from MCBrain2layer import*

##consts###
MAXTOP=5
MAXBOT=12
chunk_size = 1000000
directory = 'InferedTrajectoriesMCBRAIN'
def load_and_concatenate(directory, prefix, num_chunks):
    concatenated_array = []
    for i in range(num_chunks):
        chunk_file = os.path.join(directory, f'{prefix}_chunk{i}.npy')
        chunk_array = np.load(chunk_file)
        concatenated_array.append(chunk_array)
    return np.concatenate(concatenated_array)
total_length = 50_000_001
# Calculate the number of chunks needed
num_chunks = total_length // chunk_size
if total_length % chunk_size != 0:
    num_chunks += 1

#############DATASETS HERE###############
#jochdata = scipy.io.loadmat('StochasticRecurrentSymmetric (1).mat')
jochdata = scipy.io.loadmat('StochasticRecurrentSymmetricNE11NR4.mat')

#Pmnop_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/MCBrain2layer.cl'

print(jochdata.keys())
Rkij = jochdata['R_kij']
Ekij = jochdata['E_kij']
set = 3
Upleft=Rkij[0,:,set]
Upright=Rkij[1,:,set]
Botleft=Ekij[0,:,set]
Botright=Ekij[1,:,set]
# jocha = np.round(Upleft[:]*4)
# jochb = np.round(Upright[:]*4)
# jochc = np.round(Botleft[:]*11)
# jochd = np.round(Botright[:]*11)
jocha = Upleft[:]/4
jochb = Upright[:]/4
jochc = Botleft[:]/11
jochd = Botright[:]/11
print(len(jochd))

#############DT = .01 FORWARDS###############################
#I1, dt.01, set15
# jochFwrseed1A15 = np.load('JochenI_1dt_.01HseedA.npy')
# jochFwrseed1B15 = np.load('JochenI_1dt_.01HseedB.npy')
# jochFwrseed1C15 = np.load('JochenI_1dt_.01HseedC.npy')
# jochFwrseed1D15 = np.load('JochenI_1dt_.01HseedD.npy')

#I.0625, dt.01,
# jochFwrseed1A0=np.load('JochenI_0625dt_.01HseedA.npy')
# jochFwrseed1B0=np.load('JochenI_0625dt_.01HseedB.npy')
# jochFwrseed1C0=np.load('JochenI_0625dt_.01HseedC.npy')
# jochFwrseed1D0=np.load('JochenI_0625dt_.01HseedD.npy')

#################DT = .001 FORWARDS###########################
#I1, dt.001,   ### 50 SETS OF 1 MILLION LENGTHS
jochFwrseed1A0 = load_and_concatenate(directory, 'JochenI_1dt_.001HseedA', num_chunks)
jochFwrseed1B0 = load_and_concatenate(directory, 'JochenI_1dt_.001HseedB', num_chunks)
jochFwrseed1C0 = load_and_concatenate(directory, 'JochenI_1dt_.001HseedC', num_chunks)
jochFwrseed1D0 = load_and_concatenate(directory, 'JochenI_1dt_.001HseedD', num_chunks)
print(len(jochFwrseed1D0))

# I 0625 dt.001   ### 50 SETS OF 1 MILLION LENGTHS
# jochFwrseed1A0 = load_and_concatenate(directory, 'JochenI_0625dt_.001HseedA', num_chunks)
# jochFwrseed1B0 = load_and_concatenate(directory, 'JochenI_0625dt_.001HseedB', num_chunks)
# jochFwrseed1C0 = load_and_concatenate(directory, 'JochenI_0625dt_.001HseedC', num_chunks)
# jochFwrseed1D0 = load_and_concatenate(directory, 'JochenI_0625dt_.001HseedD', num_chunks)

# I .375 dt.001   ### 50 SETS OF 1 MILLION LENGTHS
# jochFwrseed1A0 = load_and_concatenate(directory, 'JochenI_375dt_.001HseedA', num_chunks)
# jochFwrseed1B0 = load_and_concatenate(directory, 'JochenI_375dt_.001HseedB', num_chunks)
# jochFwrseed1C0 = load_and_concatenate(directory, 'JochenI_375dt_.001HseedC', num_chunks)
# jochFwrseed1D0 = load_and_concatenate(directory, 'JochenI_375dt_.001HseedD', num_chunks)

# I .6875 dt.001   ### 50 SETS OF 1 MILLION LENGTHS
# jochFwrseed1A0 = load_and_concatenate(directory, 'JochenI_6875dt_.001HseedA', num_chunks)
# jochFwrseed1B0 = load_and_concatenate(directory, 'JochenI_6875dt_.001HseedB', num_chunks)
# jochFwrseed1C0 = load_and_concatenate(directory, 'JochenI_6875dt_.001HseedC', num_chunks)
# jochFwrseed1D0 = load_and_concatenate(directory, 'JochenI_6875dt_.001HseedD', num_chunks)

def gammaDist(dataA, dataB):
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
def lowerDist(dataC):
    CDDist=np.zeros(MAXBOT)
    for i in range(len(dataC)):
        c = int(dataC[i])
        CDDist[c]+=1
    CDDist/=np.sum(CDDist)
    return CDDist
def combinedDists(dataA,dataC,dataD):
    ACDist=np.zeros((MAXTOP,MAXBOT))
    ADDist=np.zeros((MAXTOP,MAXBOT))
    for i in range(len(dataC)):
        a = int(dataA[i])
        c = int(dataC[i])
        d = int(dataD[i])

        ACDist[a,c]+=1
        ADDist[a,d]+=1

    ACDist/=np.sum(ACDist)
    ADDist/=np.sum(ADDist)

    return ACDist, ADDist
def ExtractDominanceStatisticsTwo(dt, R_ki):
    NR = 4
    Rx_ki = (R_ki == np.uint8(NR)).astype(np.uint8)
    Rx_ki = Rx_ki[:, np.sum(Rx_ki, axis=0) == 1]

    D_i = Rx_ki[0, :] + 2 * Rx_ki[1, :]
    Ni = len(D_i)

    ti = dt * np.arange(Ni)

    dix = np.where(D_i[:-1] != D_i[1:])[0]

    if dix.size > 0:
        flag = D_i[0]

        if flag == 1:
            tD1on = [ti[0]]
        else:
            tD1on = []
        tD1off = []

        if flag == 2:
            tD2on = [ti[0]]
        else:
            tD2on = []
        tD2off = []

        tDXon = [ti[0]]
        tDXoff = []

        List = [flag]

        for i in range(Ni - 1):
            precD = D_i[i]
            succD = D_i[i + 1]

            if precD != succD:
                if precD == 1:
                    tD1off.append(ti[i])
                elif precD == 2:
                    tD2off.append(ti[i])
                tDXoff.append(ti[i])

                if succD == 1:
                    tD1on.append(ti[i])
                    flag = 1
                elif succD == 2:
                    tD2on.append(ti[i])
                    flag = 2
                tDXon.append(ti[i])

                List.append(flag)

        if flag == 1:
            tD1off.append(ti[-1])
        elif flag == 2:
            tD2off.append(ti[-1])
        tDXoff.append(ti[-1])

        N1 = min(len(tD1on), len(tD1off))
        N2 = min(len(tD2on), len(tD2off))

        D1 = np.array(tD1off) - np.array(tD1on)
        D2 = np.array(tD2off) - np.array(tD2on)
        DX = np.array(tDXoff) - np.array(tDXon)

        Mu1_1 = np.sum(D1) / N1
        Mu1_2 = np.sum(D2) / N2

        Mu2_1 = np.sum((D1 - Mu1_1) ** 2) / N1
        Mu2_2 = np.sum((D2 - Mu1_2) ** 2) / N2

        Mu3_1 = np.sum((D1 - Mu1_1) ** 3) / N1
        Mu3_2 = np.sum((D2 - Mu1_2) ** 3) / N2

        Std_1 = np.sqrt(Mu2_1)
        Std_2 = np.sqrt(Mu2_2)

        RevCount = N1 + N2
        Duration = [Mu1_1, Mu1_2]
        Variance = [Mu2_1, Mu2_2]
        CV = [Std_1 / Mu1_1, Std_2 / Mu1_2]
        Skewness = [Mu3_1 / Mu2_1 ** 1.5, Mu3_2 / Mu2_2 ** 1.5]

        ix1 = np.where(np.array(List) == 1)[0]
        ix2 = np.where(np.array(List) == 2)[0]

        DcorrX = DX.copy()
        DcorrX[ix1] = (DX[ix1] - Mu1_1) / Std_1
        DcorrX[ix2] = (DX[ix2] - Mu1_2) / Std_2

        R11 = np.corrcoef(DcorrX[:-1], DcorrX[1:])
        R12 = np.corrcoef(DcorrX[:-2], DcorrX[2:])
        R13 = np.corrcoef(DcorrX[:-3], DcorrX[3:])

        Sequential = [R11[0, 1], R12[0, 1], R13[0, 1]]
    else:
        RevCount = 0
        Duration = [np.inf, np.inf, np.inf]
        Variance = [0, 0, 0]
        CV = [1, 1, 1]
        Skewness = [1, 1, 1]
        Sequential = [0, 0, 0]

    return Duration, Variance, CV, Skewness, Sequential, RevCount
def ExtractTransitionIntervals(NA,NC):
    transupA=np.zeros(5)
    transdownA = np.zeros(5)
    transupC = np.zeros(11)
    transdownC = np.zeros(11)

    for i in range(len(NA)-1):
        print(i)
        dA = int(NA[i]-NA[i+1])
        dC = int(NC[i]-NC[i+1])
        if dA>=0:
            transdownA[dA]+=1
        elif dA<0:
            transupA[-1*dA]+=1
        if dC>=0:
            transdownC[dC]+=1
        elif dC<0:
            transupC[-1*dC]+=1
    return transdownA, transupA, transdownC, transupC

#transa, transc = ExtractTransitionIntervals(jochFwrseed1A0,jochFwrseed1C0)
#transdownAjoch, transupAjoch, transdownCjoch, transupCjoch = ExtractTransitionIntervals(jocha,jochc)

#print(transa,transc,'deltaR=0, deltaR =1, deltaR>=2 MC MODEL NR =4 NE = 1')
# print(transdownAjoch,'transdownA\n')
# print(transupAjoch,'transupA\n')
# print(transdownCjoch,'transdownC\n')
# print(transupCjoch,'transupC\n')


#DurationJoch, VarJoch, CVJoch,SkewnessJoch,CCsJoch,RevCJoch =ExtractDominanceStatisticsTwo(.001,(jocha,jochb))
#Durationfwr, Varfwr, CVfwr, Skewnessfwr, CCsfwr, RevCfwr = ExtractDominanceStatisticsTwo(.001, (jochFwrseed1A0, jochFwrseed1B0))
#Durationfwr, Varfwr, CVfwr, Skewnessfwr, CCsfwr, RevCfwr = ExtractDominanceStatisticsTwo(.01, (jochFwrseed1A15, jochFwrseed1B15))

#NA4Stronglowertrans = lowerTransitions(4,boundStrong)
# print('durations: joch, fwr', DurationJoch, Durationfwr)
# print('variance: joch, fwr', VarJoch, Varfwr)
# print('CV: joch, fwr', CVJoch, CVfwr)
# print('Skewness: joch, fwr', SkewnessJoch, Skewnessfwr)
# print('CCs: joch, fwr', CCsJoch, CCsfwr)
# print('RevCs: joch, fwr', RevCJoch, RevCfwr)



#GET ALL DISTS
toggletimesjoch = gammaDist(jocha,jochb)
#toggletimesfrwseed1 = gammaDist(jochFwrseed1A15,jochFwrseed1B15)
toggletimesfrwseed1 = gammaDist(jochFwrseed1A0,jochFwrseed1B0)
print(np.sum(toggletimesjoch),np.sum(toggletimesfrwseed1))

np.save('toggletimesjochset15',toggletimesjoch)
#np.save('toggletimesfwrseed1',toggletimesfrwseed1)
#np.save('toggletimesfwrseed1set0',toggletimesfrwseed1)
np.save('toggletimesfwrseed1set15ub',toggletimesfrwseed1)

# LowerJoch = lowerDist(jochc)
# #LowerFwrseed1 = lowerDist(jochFwrseed1C15)
# LowerFwrseed1 = lowerDist(jochFwrseed1C0)
#
# ACJoch, ADJoch = combinedDists(jocha,jochc,jochd)
# #ACFwrSeed1,ADFwrSeed1= combinedDists(jochFwrseed1A15,jochFwrseed1C15,jochFwrseed1D15)
# ACFwrSeed1,ADFwrSeed1= combinedDists(jochFwrseed1A0,jochFwrseed1C0,jochFwrseed1D0)

###########PLOTTTING###########


#gamma plots
nbins=120
min_bin = min(min(toggletimesfrwseed1), min(toggletimesjoch))
max_bin = max(max(toggletimesfrwseed1), max(toggletimesjoch))
bins = np.linspace(min_bin, max_bin, nbins)

plt.hist(toggletimesjoch,bins=bins)
plt.title('jochens flip distribution')
plt.xlim((0,8000))
# plt.ylim((0,3500))
plt.figure()

plt.hist(toggletimesfrwseed1,bins=bins)
plt.title('joch infered seed 1 flip distribution')
plt.xlim((0,8000))
# plt.ylim((0,3500))

plt.show()

#1D distribution of lowers
numsB= np.arange(MAXBOT)

plt.figure()
plt.bar(numsB,LowerJoch)
plt.title('C Activity Jochen Distribution')
plt.figure()

plt.bar(numsB,LowerFwrseed1)
plt.title('C Activity joch infered seed1 Distribution')
plt.show()

##heatmaps
#AC dists
plt.figure()
plt.imshow(ACJoch, cmap='viridis', interpolation='nearest')
plt.xlabel('#C')
plt.ylabel('#A')
plt.title('joch A-C heatmap')

plt.figure()
plt.imshow(ACFwrSeed1, cmap='viridis', interpolation='nearest')
plt.xlabel('#C')
plt.ylabel('#A')
plt.title('infered seed1 A-C heatmap')
plt.show()
####AD distributions
plt.figure()
plt.imshow(ADJoch, cmap='viridis', interpolation='nearest')
plt.xlabel('#D')
plt.ylabel('#A')
plt.title('joch A-D heatmap')

plt.figure()
plt.imshow(ADFwrSeed1, cmap='viridis', interpolation='nearest')
plt.xlabel('#D')
plt.ylabel('#A')
plt.title('infered seed1 A-D heatmap')
plt.show()
