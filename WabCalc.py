import numpy as np
from scipy.special import iv


l = 3.8
e = .2
kt = .582


acidparams = np.zeros((20,2))
acidparams[0]=(5.04,.730)
acidparams[1]=(6.56,0)
acidparams[2]=(5.68,.432)
acidparams[3]=(5.58,.378)
acidparams[4]=(5.48,.595)
acidparams[5]=(6.02,.514)
acidparams[6]=(5.92,.459)
acidparams[7]=(4.5,.649)
acidparams[8]=(6.08,.514)
acidparams[9]=(6.18,.973)
acidparams[10]=(6.18,.973)
acidparams[11]=(6.36,.514)
acidparams[12]=(6.18,.838)
acidparams[13]=(6.36,1)
acidparams[14]=(5.56,1)
acidparams[15]=(5.18,.595)
acidparams[16]=(5.62,.676)
acidparams[17]=(6.78,.946)
acidparams[18]=(6.46,.865)
acidparams[19]=(5.86,.892)

def calcWab(a,b):
    lam= (acidparams[a][1]+acidparams[b][1])/2
    sig = (acidparams[a][0]+acidparams[b][0])/2
    z=e*lam/(2*kt)
    btstar= (np.sqrt(2)*np.pi *z) * np.exp(z)*(iv(-3/4,z)+iv(3/4,z)-iv(1/4,z)-iv(-1/4,z))
    wab = (4*np.pi/3) *(sig/l)**3 * btstar
    if z == 0:
        wab = 0
    return wab

wab = np.zeros((20,20))
for i in range(len(acidparams)):
    for j in range(len(acidparams)):
        wab[i][j] = calcWab(i,j)
print(wab[0][0])















