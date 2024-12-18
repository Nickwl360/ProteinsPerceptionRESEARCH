from rgRPA_allfns import *
import numpy as np
from matplotlib import pyplot as plt
from rgRPA_init import phiS, scale_init,epsilon,qc,N,scale_final

if __name__ == "__main__":
    minY = Yc*.8

    print('looping from ', Yc, 'to ', minY)

    spins, phis,Ys = getBinodal(Yc, phiC, minY)

    ###################PLOTTING########################
    plt.plot(phis, Ys, label='Binodal')
    plt.plot(spins,Ys,label='Spinodal')
    plt.xlim((0,.25))

    plt.legend()
    plt.show()