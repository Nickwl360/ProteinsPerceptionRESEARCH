from FH_PSModel_Functs import get_critical_vals, getbinodal
from matplotlib import pyplot as plt
import numpy as np

if __name__ =='__main__':
    seqNames = ['WT', 'aro+','aro-', 'aro--']
    N = 137
    #omega2List = [-0.31350964,-0.32084423,-0.28773427,-0.27752626]  #mike list
    omega2List = [-0.3479,-0.38978,-0.29146,-0.22101]                #lili List

    chiList = [(1-w2)/2 for w2 in omega2List]

    phiCtCList = [(get_critical_vals(chi),chi) for chi in chiList]

    tcNorm = phiCtCList[0][0][1]
    print(tcNorm)

    binodalList = [(getbinodal(tc,phic,chi),seqName) for (((phic,tc), chi),seqName) in zip(phiCtCList, seqNames)]

    TcList = [tc for ((phic,tc),chi) in phiCtCList]
    TcList /= tcNorm
    print(TcList)

    tcDataApprox=[332/332, (85+273.15)/332,  299.32/332, (-5+273.15)/332]
    print(tcDataApprox)

    for ((bis, spins, ts),seq) in binodalList:
        plt.plot(bis, ts/tcNorm, label=seq)

    plt.legend()
    plt.show()

