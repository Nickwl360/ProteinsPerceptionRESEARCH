from CurrentProjects.PS_ChiFitting_and_ML.FH_PSModel_Functs import get_critical_vals, getbinodal
from matplotlib import pyplot as plt
import numpy as np

setTest = 0

#### A1 - LCD ### vs Aromatic Varients ###
if setTest ==0:
    seqNames = ['WT', 'aro+','aro-', 'aro--']
    N = 137
    omega2List = [-0.3479,-0.38978,-0.29146,-0.22101] #lili predicted List


    tcDataApprox=[332/332, (85+273.15)/332,  299.32/332, (-5+273.15)/332] ### Fig3.B experimental Tc's
    print('T/Tc',tcDataApprox)

#### More A1 - LCD Alex SI predictions ####
elif setTest ==1:
    seqNames = ['WT', '+7f -7Y', '-12F +12Y']
    N = 137
    omega2List = [-0.3479, -0.312672, -0.38825116]  # lili predicted List

elif setTest ==2:
    seqNames = ['FUS-WT', 'FUS-6E', 'FUS-12E']
    N = 163
    omega2List = [-0.23405018, -0.26151404, -0.3302501]  # lili predicted List

elif setTest ==3:
    seqNames = ['RLP: Y=0.13, V=0.0', 'RLP: Y=0.1, V=0.03', 'RLP: Y=0.06, V=0.06']
    N = 326
    omega2List = [-0.27687773, -0.22338626, -0.14956477]  # lili predicted List

elif setTest ==4:
    print('FH PHASE DIAGRAMS FOR TDP-43 LOW COMPLEXITY DOMAIN & VARIANTS')
    seqNames=['WT','VLIM-F','W385G','FYW-L','VLIM-S','F-S','FYW-S']
    N = 152
    omega2List = [-0.1488061,-0.21055762,-0.13680525,-0.08521357,-0.13947238,-0.089303896,-0.053069234]

if __name__ =='__main__':

    chiList = [(1-w2)/2 for w2 in omega2List]
    phiCtCList = [(get_critical_vals(chi,N),chi) for chi in chiList]

    tcNorm = phiCtCList[0][0][1]
    print(tcNorm)
    print(N)
    binodalList = [(getbinodal(tc,phic,chi,N),seqName) for (((phic,tc), chi),seqName) in zip(phiCtCList, seqNames)]

    TcList = [tc for ((phic,tc),chi) in phiCtCList]
    TcList /= tcNorm
    print('T/Tc',TcList)



    for ((bis, spins, ts),seq) in binodalList:
        plt.plot(bis, ts/tcNorm, label=seq)
        plt.ylim(0.8,1.1)
        plt.ylabel(r'$T^*$')
        plt.xlabel(r'$\phi$')


    plt.legend()
    plt.savefig('FH_PhaseDiagrams/AlexSI_TDP43_LCD_fromw2Pred10-3-24')
    plt.show()

