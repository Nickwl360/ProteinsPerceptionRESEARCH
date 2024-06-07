from fgRPAFuncts import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import least_squares

#CONSTANTS##################################################################
phiS = 0.01
seqs = getseq('SCDtests.xlsx')
seq1 = 'MGDEDWEAEINPHMSSYVPIFEKDRYSGENGDNFNRTPASSSEMDDGPSRRDHFMKSGFASGRNFGNRDAGECNKRDNTSTMGGFGVGKSFGNRGFSNSRFEDGDSSGFWRESSNDCEDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFGSRRPVLSGTGNGDTSQSRSGSGSERGGYKGLNEEVITGSGKNSWKSEAEGGES'
qs = getcharges(seq1)
qc = abs(sum(qs))/N
#print(qc)
resolution = .0005
epsilon = 1e-7
phiC,Yc = findCrits(phiS)
minY = Yc*.9
print('looping from ', Yc, 'to ', minY)


def findSpinlow(Y,phiC):
    initial = phiC-epsilon
    bounds = [(epsilon, phiC-epsilon)]
    result = minimize(FreeEnergyD2reverse, initial, args=(Y,phiS,),method='SLSQP',bounds=bounds)
    #result = fsolve(FreeEnergyD2reverse, x0=initial, args=(Y,phiS))
    return result
def findSpinhigh(Y,phiC):
    initial = phiC+epsilon
    bounds = [(phiC+epsilon, 1-epsilon)]
    result = minimize(FreeEnergyD2reverse, initial, args=(Y,phiS),method='SLSQP',bounds=bounds)
    #result = fsolve(FreeEnergyD2reverse, x0=initial, args=(Y,phiS))
    return result
def FreeEnergyD2reverse(phiM,Y,phiS):
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
    result = integrate.quad(felD2integrand,lowerlim,upperlim,args=(Y,phiM,),limit=150)
    d2fel= result[0]/(4*np.pi*np.pi)
    return np.sqrt((d2s + d2fel)**2)
    #return d2s + d2fel

def TotalFreeEnergy(variables,Y):
    phi1,phi2 = variables
    v = (phiC-phi2)/(phi1-phi2)
    eqn = v * ftot(phi1,Y,phiS) + (1-v)*ftot(phi2,Y,phiS)
    return eqn

def findPhisnoconst(Y,phiC,lastphi1,lastphi2):
    phi1spin = findSpinlow(Y,phiC)[0]
    print('next')
    phi2spin= findSpinhigh(Y,phiC)[0]
    print(lastphi1, 'last')
    print(Y,phi1spin,phi2spin, 'starting spinpoints')
    if lastphi1==phiC:
        bounds = [(phi1spin*.8, phi1spin - epsilon), (phi2spin + epsilon, phi2spin*1.2)]
        initial_guess = (phi1spin*.9,phi2spin*1.1)
    else:
        bounds = [(lastphi1*.5,lastphi1-epsilon), (lastphi2+epsilon,lastphi2*1.5)]
        initial_guess = (lastphi1-epsilon,lastphi2+epsilon)
    maxL = minimize(TotalFreeEnergy, initial_guess, args=(Y,), method='Nelder-Mead', bounds=bounds)
    maxparams = maxL.x
    return maxparams

def getbinodal(Yc,phiC):
    phibin=np.full(1,phiC)
    Ybin =np.full(1,Yc)
    Ytest=Yc-resolution
    save1,save2= phiC[0],phiC[0]
    tries = 100
    take = 0
    while take>tries:
        if Ytest== Yc-resolution:
            phil,phi2 = findPhisnoconst(Ytest,phiC,phiC,phiC)
            save1,save2 = phil,phi2
        else:
            phil,phi2 = findPhisnoconst(Ytest,phiC,save1,save2)
            save1,save2 = phil,phi2
        print(Ytest, phil, phi2, 'results. loop # = ', take)
        phibin = np.concatenate(([phil], phibin, [phi2]))
        Ybin = np.concatenate(([Ytest], Ybin, [Ytest]))
        Ytest-=resolution
        take +=1

    return phibin,Ybin

print(phiC,Yc)
phis,chis = getbinodal(Yc,phiC)
plt.plot(phis, chis, label='Binodal')

phi_values = np.linspace(.001, .4, 1000)
Ts = []
for i in phi_values:
    Y = fsolve(FreeEnergyD2,x0=.4,args=(i,phiS))
    Ts.append(Y)

plt.plot(phi_values,Ts,label='Spinodal')
plt.legend()
plt.show()