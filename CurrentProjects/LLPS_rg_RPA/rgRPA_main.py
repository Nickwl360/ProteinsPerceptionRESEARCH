from rgRPA_allfns import *
import numpy as np
from matplotlib import pyplot as plt
from rgRPA_init import phiS, scale_init,epsilon,qc,N,scale_final

if __name__ == "__main__":
    minY = Yc*.98
    print('looping from ', Yc, 'to ', minY)

    phis,chis = getBinodal(Yc, phiC, minY)
    phiMs = np.linspace(1e-3, .15, 40)
    Ys = getSpinodalrg(phiMs)

    ###################PLOTTING########################
    plt.plot(phis, chis, label='Binodal')
    plt.plot(phiMs,Ys,label='Spinodal')
    plt.xlim((0,1))

    plt.legend()
    plt.show()