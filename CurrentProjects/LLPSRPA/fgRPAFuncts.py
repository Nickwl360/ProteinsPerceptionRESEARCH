import numpy as np
from OldProteinProjects.SCDcalc import *
from scipy import integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root_scalar

#CONSTANTS##################################################################
#phiS = 0.01
phiS = 0
seqs = getseq('../../OldProteinProjects/SCDtests.xlsx')
ddx4n1 = 'MGDEDWEAEINPHMSSYVPIFEKDRYSGENGDNFNRTPASSSEMDDGPSRRDHFMKSGFASGRNFGNRDAGECNKRDNTSTMGGFGVGKSFGNRGFSNSRFEDGDSSGFWRESSNDCEDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFGSRRPVLSGTGNGDTSQSRSGSGSERGGYKGLNEEVITGSGKNSWKSEAEGGES'
ddx4n1CS = 'MGDRDWRAEINPHMSSYVPIFEKDRYSGENGRNFNDTPASSSEMRDGPSERDHFMKSGFASGDNFGNRDAGKCNERDNTSTMGGFGVGKSFGNEGFSNSRFERGDSSGFWRESSNDCRDNPTRNDGFSDRGGYEKGNNSEASGPYERGGRGSFDGCRGGFGLGSPNNRLDPRECMQRTGGLFGSDRPVLSGTGNGDTSQSRSGSGSERGGYKGLNEKVITGSGENSWKSEARGGES'
IP5 = 'HAQGTFTSDKSKYLDERAAQDFVQWLLDGGPSSGAPPPS'

seq1 = IP5
# N = len(seq1)
# qs = getcharges(seq1)
# qc = abs(sum(qs))/N
#print(qc)

def pH_qs(seq, ph):
    charges = []
    # get charge array
    for letter in seq:
        if letter == 'E' or letter == 'D':
            if letter == 'E':
                q = -1*(10**(-1*(4.15- ph)))/(1+ 10**(-1*(4.15-ph)))
            elif letter == 'D':
                q = -1 * (10 ** (-1*(3.71 - ph))) / (1 + 10 ** (-1*(3.71 - ph)))
            charges.append(q)
        elif letter == 'R' or letter == 'K':
            if letter == 'R':
                q = (10**(12.1- ph))/(1 + 10**(12.1-ph))
            elif letter == 'K':
                q = (10**(10.67- ph))/(1 + 10**(10.67-ph))
            charges.append(q)
        else:
            charges.append(0)
    return charges
N = len(seq1)
qs = pH_qs(seq1,7.42)
qc = abs(sum(qs))/N
#qc = sum(qs)/N


print(qc)
#FUNCTIONS#################################################################
def getSigShift(qs):
    sigS = []
    for i in range(len(qs)-1):
        sigi=0
        for j in range(len(qs)-1):
            if (j+i)<= len(qs)-1:
                sigi += qs[j]*qs[j+i]
        if i ==0:
            sigS.append(sigi)
        if i!=0:
            sigS.append(2*sigi)
    return sigS

sigShift = getSigShift(qs)

def xee(k,sigS):
    xeesum=0
    for i in range(0,N-1):
        xeesum += sigS[i]*np.exp((-1/6)*(k*k)*i)
    return xeesum/N
def FreeEnergyD2(Y,phiM,phiS):
    d2s =0
    #################Entropyd2##########
    if phiM!=0:
        if phiM != (phiS-1)/(-1*qc -1):
            d2s = 1/(N*phiM)+ qc/phiM + ((-qc-1)**2)/(-1*qc*phiM-phiS-phiM + 1)
        elif phiM == (phiS-1)/(-1*qc -1):
            d2s = qc/phiM + 1/(N*phiM)
    else: d2s = (-1*qc -1)**2/(-1*phiS +1)

    #################Electrofreeenergyd2###########
    upperlim = np.inf
    lowerlim = 0
    print('attempt')
    result = integrate.quad(felD2integrand,lowerlim,upperlim,args=(Y,phiM), limit=150)
    d2fel= result[0]/(4*np.pi*np.pi)
    return d2s + d2fel
def felintegrand(k,Y,phiM,phiS):
    x = xee(k,sigS=sigShift)
    return (k**2)*np.log(1 + ((4*np.pi)/(k**2*Y))*(phiS+(qc+x)*phiM))
def felD2integrand(k,Y,phiM):
    x = xee(k,sigS=sigShift)
    return -1 * k * k * (4 * np.pi / Y)**2 * (qc + x)**2 / \
           (((4 * np.pi / Y) * (phiS + (qc + x) * phiM) + (k * k))**2)

def fel(phiM, Y, phiS):
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(felintegrand, lowerlim, upperlim, args=(Y, phiM,phiS), limit=150)
    fel = result[0] / (4 * np.pi * np.pi)
    return fel

def Entropy(phiM,phiS):
    phiC = qc*phiM                 ##solve eqn for phiC
    phiW = 1-phiM-phiS-phiC
    #################FIGURE OUT LOGIC FOR 0s
    if phiS!= 0:
        return (phiM/N)*np.log(phiM) + phiS* np.log(phiS) + phiC*np.log(phiC) + phiW*np.log(phiW)
    else:return (phiM/N)*np.log(phiM)+ phiC*np.log(phiC) + phiW*np.log(phiW)

def ftot(phiM,Y,phiS):
    return Entropy(phiM,phiS) + fel(phiM,Y,phiS)
def getSpinodal(phiMs):
    Ys=[]
    for j in range(len(phiMs)):
        i = phiMs[j]
        y = root_scalar(FreeEnergyD2,args=(i,phiS,),x0=.5, bracket=[1e-3,.99])
        print(i,y.root)
        Ys.append(y.root)
    return Ys
def SpinodalY(phiM,phiS):
    guess1 = .5
    y = fsolve(FreeEnergyD2, args=(phiM, phiS,), x0=guess1)
    print(phiM, y)
    return -1*y
def findCrits(phiS):
    phiC=0
    Yc=0
    bounds = [(.001,.49)]
    Yc = minimize(SpinodalY, x0=.09, args=(phiS,),method='Nelder-Mead', bounds=bounds)
    phiC = Yc.x
    return phiC, -1*Yc.fun

if __name__ == "__main__":
    x,y = findCrits(phiS)
    print(x,y, ' FINAL phiC/Yc')
    ######################FINDING SPINODAL#########################
    phiMs = np.linspace(1e-3,.799,100)
    #Ys= [0.28270371056883575,0.6183707670808203,0.6644949264903841,0.663421867142575,0.6479709933346601, 0.6278191295235688,0.6065203753916169,0.5855094630020322,0.5653729080073601,0.5463272614483451,0.5284200885862346,0.5116210713183862,0.4958654699978174,0.4810755008183442,0.46717095218422255, 0.45407436517366956,0.4417134029427058,0.43002175146490546,.4189,.4084,.3984,.3888,.3797,.3709,.36255,.35449,.34673,.33924,.3320,.3250,.31827,.3117,.3053,.2991,.2931,.287265,.2815,.2759,.2705,.26516,.2599,.2548,.24485,.2400,.2352,.23056,.2259,.2214,.2169,.21250,.208139,.20382,.199567,.19535,.19118,.18705,.1829,.1789,.1748,.17088,.16691,.1629,.1590,.1551,.15126,.14739,.14353,.13969,.13585,.13202,.12819,.12436,.1205,.116689,.1128,.10898,.1051,.10121,.0972,.0933,.08936,.08534,.08128,.077162,.07298,.068729,.06439,.059957,.055403,.0507,.04583,.04075,.035387,.02964,.02333,.016100,.0065,0,0,0]
    Ys = getSpinodal(phiMs)

    ##################PLOT SPINODAL######################
    plt.figure()
    plt.plot(phiMs,Ys)
    plt.xlabel('Volume Fraction of Protein (phi)')
    plt.ylabel('l/lB')
    plt.title(('L/Lb vs Volume Fraction Phase Diagram DDx4, SPINODAL'))
    plt.show()



