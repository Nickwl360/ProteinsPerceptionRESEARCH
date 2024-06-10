from CurrentProjects.PerceptionE25R5.MCBrain2layer import *
from MaxCalBrainTest1 import *


def getphase1(maxe,size):
    halphavalues = np.linspace(0,maxe,size)
    hbetavalues = np.linspace(0,maxe,size)
    phasediagram = np.zeros((len(halphavalues),len(hbetavalues)))
    for i in range(len(halphavalues)):
        for j in range(len(hbetavalues)):
            print(i,j)
            phasediagram[i][j] = calcaminusb1(halphavalues[i],hbetavalues[j])
    plt.matshow(phasediagram, origin='lower', extent=(min(halphavalues), max(halphavalues), min(hbetavalues), max(hbetavalues)))
    plt.colorbar(label='Value')
    plt.xlabel('halpha')
    plt.ylabel('hbeta')
    plt.title('(<NA> - <NB>)/MAXTOP:  kcoop=.1, kcomp = .1')
    plt.show()
    return

# getphase1(1,20)
#
def getphase2(maxe,size):
    hgammavalues = np.linspace(0,maxe,size)
    hdeltavalues = np.linspace(0,maxe,size)
    phasediagram = np.zeros((len(hgammavalues),len(hdeltavalues)))
    for i in range(len(hgammavalues)):
        for j in range(len(hdeltavalues)):
            phasediagram[i][j] = calcaminusb(hgammavalues[i],hdeltavalues[j])
    plt.matshow(phasediagram, origin='lower', extent=(min(hgammavalues), max(hgammavalues), min(hdeltavalues), max(hdeltavalues)))
    plt.colorbar(label='Value')
    plt.xlabel('hgamma')
    plt.ylabel('hdelta')
    plt.title('(<NA> - <NB>)/MAXTOP')
    plt.show()
    return
getphase2(1,10)
