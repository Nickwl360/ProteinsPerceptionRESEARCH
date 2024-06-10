from CurrentProjects.PerceptionE25R5.MaxCalBrainTest1 import*
import numpy as np
import matplotlib.pyplot as plt
Pijkl_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/brainpijkl.cl'


def calcdecision(dataA,dataB):
    SUMA= 0
    SUMB=0
    for i in range(len(dataA)):
        SUMA+= dataA[i]/25
        SUMB+= dataB[i]/25
    SUMA/=len(dataA)
    SUMB/=len(dataB)
    return abs(SUMA-SUMB)

halphas = np.linspace(0,1,25)
hbetas = np.linspace(0,1,25)
start = [0,0]
R = np.zeros((25,25))

for i in range(len(halphas)):
    for j in range(len(hbetas)):
        params = (halphas[i], .01, hbetas[j], .01 +(halphas[i]-hbetas[j]), .1, .1)
        p = run_program(params,Pijkl_prog)
        pequ = getequilibrium(p)
        As,Bs = simulation(start,params)
        R[i][j] = calcdecision(As,Bs)
        print(i,j,R[i][j])

plt.imshow(R)
plt.show()
