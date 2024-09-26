from perception_TijCalc import runtime_program, renormalize
from perception_Traj_init import *
import numpy as np

#start state 0-5 for each
Ri = 4
Riprime = 1

n_space = np.arange(1,10000)

Pi = (Ri, Riprime)


tij_gpu = runtime_program(params,(ll1,ll2),prog_path)
tij_reshape = tij_gpu.reshape((MAXTOP,MAXTOP,MAXTOP,MAXTOP))
tij_normalized = renormalize(tij_reshape)


## want (5,5,5,5) X (5,5)
Pj_nt = np.zeros((MAXTOP,MAXTOP))
Pj_nt[1][0]= 1
print(Pj_nt)

def calcPjn(Pi,Tijn):
    pjn = np.zeros((MAXTOP,MAXTOP))
    for naj in range(MAXTOP):
        for nbj in range(MAXTOP):
            for i in range(MAXTOP):
                for j in range(MAXTOP):
                    pjn[naj][nbj] += Tijn[i][j][naj][nbj]
    return pjn

def getPj_nt(Pi, Tij,nspace):

    #start with delta function at Pi#
    nai,nbi = Pi
    Pj_nt0 = np.zeros((MAXTOP, MAXTOP))
    Pj_nt0[nai][nbi] = 1

    Pj_nt_list = []
    Pj_nt_list.append(Pj_nt0)

    #iterate according to Tij^n#
    for n in nspace:
        Tijn = np.linalg.matrix_power(Tij,n)
        #Pj_nt= calcPjn(Pj_nt0,Tijn)
        #Tijn.transpose()
        Pj_nt = np.einsum('ijkl,kl->ij',Tijn, Pj_nt0)
        #Pj_nt =  Pj_nt0 @ Tijn

        #print(np.shape(Pj_nt))
        totalP = np.sum(Pj_nt)
        Pj_nt/=totalP

        Pj_nt_list.append(Pj_nt)
    return Pj_nt_list

Pj_nttraj = getPj_nt(Pi,tij_normalized,n_space)

def get_avgN_t(Pj_nt,n_space,Pi):
    na_indices = np.arange(MAXTOP).reshape(-1,1)
    nb_indices = np.arange(MAXTOP)
    print(na_indices,nb_indices)
    nai,nbi = Pi
    avgNa_t, avgNb_t = [nai],[nbi]
    for n in n_space:
        avgNa_t.append(np.sum(na_indices*Pj_nt[n]))
        avgNb_t.append(np.sum(nb_indices*Pj_nt[n]))
    return avgNa_t, avgNb_t

avgNa_list, avgNb_list = get_avgN_t(Pj_nttraj,n_space,Pi)

print(avgNa_list,'\n')
print(avgNb_list)