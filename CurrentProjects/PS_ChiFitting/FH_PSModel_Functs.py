import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from ChiFit_Main import *
from scipy.optimize import brenth, brent


def critical_estimate(N):
    phiC = (1 / (1 + np.sqrt(N)))
    chiC = .5*(1 + 1/np.sqrt(N))**2
    return phiC, chiC

def FH_free_energy(chi, phi):
    return (1/NA)* phi * np.log(phi) + (1 - phi)*(1/NB) * np.log(1 - phi) + chi * phi * (1 - phi)
def FH_Seperated(variables,chi):
    phi1,phi2 = variables
    phiC, XC = critical_estimate(NA)
    v = (phiC-phi2)/(phi1-phi2)
    eqn = v * FH_free_energy(chi, phi1) + (1 - v) * FH_free_energy(chi, phi2)
    return eqn
def d2_FH(chi,phi):
    return (1 / (NA * phi)) + (1 / (NB * (1 - phi))) - 2 * chi


def get_spins(chi,phiC):
    #phiMax = (1-2*phiS)/(1+qc)-epsilon
    phiMax = 1-epsilon
    phi1 = brenth(d2_FH, epsilon, phiC, args=(chi,))
    phi2 = brenth(d2_FH, phiC, phiMax,args = (chi,))
    return phi1,phi2


def findPhisnoconst(chi,phiC):
    phi1spin,phi2spin=get_spins(chi,phiC)
    print(chi,phi1spin,phi2spin)
    bounds = [(epsilon,phi1spin-epsilon), (phi2spin+epsilon,1-epsilon)]
    # bounds = [(epsilon,phiC-epsilon), (phiC+epsilon,1-epsilon)]
    initial_guess = (phi1spin*.9,phi2spin+epsilon)
    # initial_guess = (epsilon,1-epsilon)

    maxL = minimize(FH_Seperated, initial_guess, args=(chi,), method='Nelder-Mead', bounds=bounds)
    maxparams = maxL.x
    return maxparams
def getbinodal(chiC,phiC):
    phibin=[phiC]
    chibin =[1/chiC]
    chitest=chiC+resolution
    while chitest<(2):
        phil,phi2 = findPhisnoconst(chitest,phiC)
        phibin = np.concatenate(([phil], phibin, [phi2]))
        chibin = np.concatenate(([1/chitest], chibin, [1/chitest]))
        chitest+=resolution

    return phibin,chibin

phic, XC = critical_estimate(N)
print(phic,XC)
phis,chis = getbinodal(XC,phic)
plt.plot(phis, chis, label='Binodal')

phi_values = np.linspace(1.0e-10, .99999, 1000)
Ts = []
for i in phi_values:
    X = fsolve(d2_FH, .5, args=(i,))
    Ts.append(1/X)

plt.plot(phi_values,Ts,label='Spinodal')
plt.legend()
plt.show()



