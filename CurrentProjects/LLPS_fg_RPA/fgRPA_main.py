from fgRPA_functions import *
import numpy as np
from matplotlib import pyplot as plt
from fgRPA_init import phiS, scale_init,epsilon,qc,N, scale_final

if __name__ == "__main__":
    minY = Yc*.90
    print('looping from ', Yc, 'to ', minY)

    phis,chis = getBinodal(Yc, phiC, minY)
    phismooth = smooth(phis)

    phismooth3 = smooth(smooth(smooth(phis)))
    phiMs = np.linspace(1e-3, .399, 100)
    Ys = getSpinodal(phiMs)

    ###################PLOTTING########################
    plt.plot(phismooth3, chis, label='Binodal')
    plt.plot(phiMs,Ys,label='Spinodal')
    plt.show()

    plt.plot(phismooth, chis, label='Binodal')
    plt.plot(phiMs,Ys,label='Spinodal')
    plt.xlim((0,1))

    plt.legend()
    plt.show()