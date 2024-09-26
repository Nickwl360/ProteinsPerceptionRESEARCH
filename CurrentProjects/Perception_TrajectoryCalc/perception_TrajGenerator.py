from perception_TijCalc import runtime_program, renormalize
from perception_Traj_init import *
import numpy as np

#start state 0-5 for each
Ri = 3
Riprime = 3

n_space = np.arange(0,500)

Pi = [Ri, Riprime]


tij_gpu = runtime_program(params,(ll1,ll2),prog_path)
tij_reshape = tij_gpu.reshape((MAXTOP,MAXTOP,MAXTOP,MAXTOP))
tij_normalized = renormalize(tij_reshape)


def getPj_t(Pi, Tij,nspace):
    Pj_t = [Pi]
    for n in nspace:
        Tijn = np.linalg.matrix_power(Tij,n)
        Pnt = np.matmul(Pi,Tijn)
        Pj_t.append(Pnt)
    return Pj_t

print(getPj_t(Pi))

