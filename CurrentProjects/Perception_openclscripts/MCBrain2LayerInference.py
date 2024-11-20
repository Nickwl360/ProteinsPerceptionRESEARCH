from CurrentProjects.PerceptionE25R5.MCBrain2layer import *
from scipy.optimize import minimize
import scipy.io

# constants
MAXTOP = 5
MAXBOT = 12

#DATA#######################################
# dataa = np.load('Atest10mil.npy')
# datab = np.load('Btest10mil.npy')
# datac = np.load('Ctest10mil.npy')
# datad = np.load('Dtest10mil.npy')
#print(np.shape(datad))
#############JOCHEN DATA ############
jochdata = scipy.io.loadmat('StochasticRecurrentSymmetric (1).mat')
print(jochdata.keys())
Rkij = jochdata['R_kij']
Ekij = jochdata['E_kij']
set = 15
Upleft=Rkij[0,:,set]
Upright=Rkij[1,:,set]
Botleft=Ekij[0,:,set]
Botright=Ekij[1,:,set]
dataa = np.round(Upleft[:]*4)
datab = np.round(Upright[:]*4)
datac = np.round(Botleft[:]*11)
datad = np.round(Botright[:]*11)


def countbrain(dataa,datab,datac,datad):
    count = np.zeros((MAXTOP, MAXTOP, MAXBOT, MAXBOT,MAXTOP, MAXTOP, MAXBOT, MAXBOT),dtype=int)
    for i in range(0, len(datab) - 1):
        count[int(dataa[i])][int(datab[i])][int(datac[i])][int(datad[i])][int(dataa[i+1])][int(datab[i + 1])][int(datac[i+1])][int(datad[i + 1])] += 1
    return count

def renormalize(Pijkl):
    normalizefactors = np.sum(Pijkl, axis=(4, 5, 6, 7))  # Sum over the end state indices
    normalizefactors[normalizefactors == 0] = 1.0
    Pijkl /= normalizefactors[:, :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    return Pijkl

# with np.printoptions(threshold=np.inf):
#     print(countmatrix(dataa,datab))

def brainlikelyhood(params7, counts):

    #ULC,LLC,epsilon2,kcoop,kcomp,kdu,kud,kx = params8
    ULC,LLC,kcoop,kcomp,kdu,kud,kx = params7
    epsilon2=0

    (halpha, ha, hgamma, hc, kcoop, kcomp, kdu, kud, kx) = (-1 * ULC , ULC , -1 * LLC + epsilon2 / 2, LLC + -1 * epsilon2 / 2, kcoop, kcomp, kdu, kud, kx)
    params = (halpha, ha, halpha, ha , hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)

    Pmnop = runtime_program(params, Pmnop_prog)
    pmnopreshape = Pmnop.reshape((MAXTOP, MAXTOP, MAXBOT, MAXBOT, MAXTOP, MAXTOP, MAXBOT, MAXBOT))
    pmnonormal = renormalize(pmnopreshape)

    nonzero_counts = counts != 0
    nonzero_pmnonormal = pmnonormal != 0

    log_values = np.zeros_like(counts, dtype=float)
    log_values[nonzero_counts & nonzero_pmnonormal] = counts[nonzero_counts & nonzero_pmnonormal] * np.log(
        pmnonormal[nonzero_counts & nonzero_pmnonormal])
    L = np.sum(log_values)

    length=len(dataa)
    val = (-1*L)/length
    print('Likelyhood: ', val)
    return val


def minlikely(counts):
    initial = (-0.825, 0.825, -0.825, 0.825, -0.29, 0.29, -0.29, 0.29, 2.1, 2.4, 0.818, 0.17, 0.436)
    #initial8=(.825,.297,0,2.1, 2.4, 0.818, 0.17, 0.436)
    initial8=(.82,.3,0,2.05, 2.41, 0.81, 0.18, 0.44)
    initial7=(.82,.3,2.05, 2.41, 0.81, 0.18, 0.44)

    maxL = minimize(brainlikelyhood, initial7, args=(counts,), method='Nelder-Mead', tol= 1e-11)
    maxparams = maxL.x
    return maxparams



###OG:          (-0.8245, 0.8245, -0.8245, 0.8245, -0.297, 0.297, -0.297, 0.297, 2.0, 2.43, 0.8175, 0.1681, 0.4359)
#TEST1 10klong: [-0.64616358  1.31201754 -0.78079768  1.02780939 -0.27288933  0.28987539 -0.27393728  0.30376723  2.08184069  2.29232193  0.85857003  0.16843132 0.29461062]
#TEST2 100klong: [-0.91284363  0.58389405 -0.97139918  0.60840935 -0.3052354   0.3128113 -0.30137881  0.30675266  1.9619026   2.3273761   0.80547735  0.16649401  0.41952298]
##8OG:: (.8245,.297,0,2.0, 2.43, 0.8175, 0.1681, 0.4359)

#TEST3 100klong 8params:[ 0.52453841  0.29611015 -0.05404986  2.21557184  2.40425912  0.8781656 0.16627437  0.32444665]
#TEST4  1MIL 8params:[8.54553390e-01, 3.03529916e-01, 4.37734403e-04, 1.97998764e+00, 2.40462233e+00, 8.10160824e-01, 1.66776958e-01, 4.27372100e-01]

###JOCH TEST 1, set 0, I = 0.0625
 # [39.8299515    4.55428585 125.0394442   47.86269474 216.5977645  -2.14126598  84.31638625]
##JOCHTEST2,Set15, I =1
#[ 5.94159351e+00 , 6.65110825e-01,  7.68154925e+00 , 4.35942704e+00 , 1.06162505e+01 ,-3.85689956e-03,  3.26665828e+00]
###JOCH BASINHOPSET15 L=~.43
#(3.21826005e+01 8.14506833e-01 1.44808149e+01 1.61197046e+01 2.43502362e+01 1.13634232e-02 1.03209929e+01)


if __name__ == "__main__":
    count = countbrain(dataa, datab, datac, datad)
    #print(-1*brainlikelyhood((0.8245, 0.297,0, 2.0, 2.43, 0.8175, 0.1681, 0.4359),count),'og')
    # print(-1*brainlikelyhood((0.52453841,0.29611015,-0.05404986,2.21557184,2.40425912, 0.8781656,0.16627437, 0.32444665),count),'infered')
    params = minlikely(count)
    print(params)

    #1millioninfered
    # params=(8.54553390e-01, 3.03529916e-01, 4.37734403e-04, 1.97998764e+00, 2.40462233e+00, 8.10160824e-01, 1.66776958e-01, 4.27372100e-01)
    #
    ##10millioninfered
    # params = (8.24669080e-01, 2.96953251e-01, 1.70822896e-03, 1.99792955e+00, 2.43295790e+00 ,8.16179382e-01, 1.68080683e-01 ,4.37223644e-01)
    #E2=0 [0.82466722 0.2969654  1.99792941 2.43295823 0.81617904 0.16807466  0.437224  ]
    #calculate13params
    # ULC,LLC,E2,kcoop,kcomp,kdu,kud,kx = params
    # params13= (-1*ULC,ULC,-1*ULC,ULC,-1*LLC +E2/2 ,LLC - E2/2 ,-1*LLC - E2/2 ,LLC + E2/2,kcoop,kcomp,kdu,kud,kx)
    # print(params)
    # print(params13)

