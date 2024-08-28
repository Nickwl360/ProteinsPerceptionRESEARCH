from rgRPA_allfns import *
import numpy as np
from matplotlib import pyplot as plt
from rgRPA_init import phiS, scale_init,epsilon,qc,N,scale_final

if __name__ == "__main__":
    minY = Yc*.75

    print('looping from ', Yc, 'to ', minY)

    spins, phis,Ys = getBinodal(Yc, phiC, minY)
    #phiMs = np.linspace(1e-3, .15, 45)
    #YsSpin = getSpinodalrg(phiMs)

    ###################PLOTTING########################
    plt.plot(phis, Ys, label='Binodal')
    plt.plot(spins,Ys,label='Spinodal')
    plt.xlim((0,.1))

    plt.legend()
    plt.show()