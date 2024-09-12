import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from chi_fit_init import *
from scipy.optimize import brenth, brent


def get_critical_vals(N):
    phiC = (1 / (1 + np.sqrt(N)))
    chiC = .5*(1 + 1/np.sqrt(N))**2
    return phiC, chiC

def FH_free_energy(chi, phi):
    return (1/N)* phi * np.log(phi) + (1 - phi)* np.log(1 - phi) + chi * phi * (1 - phi)

def FH_Seperated(variables,chi):
    phi1,phi2 = variables
    phiC, XC = get_critical_vals(N)
    v = (phiC-phi2)/(phi1-phi2)
    eqn = v * FH_free_energy(chi, phi1) + (1 - v) * FH_free_energy(chi, phi2)
    return eqn
def d2_FH(phi, chi):
    return (1 / (N * phi)) + (1 / (1 - phi)) - 2 * chi


def get_spins(chi,phiC):
    #phiMax = (1-2*phiS)/(1+qc)-epsilon
    phiMax = 1-epsilon
    phi1 = brenth(d2_FH, epsilon, phiC, args=(chi,))
    phi2 = brenth(d2_FH, phiC, phiMax,args = (chi,))
    return phi1,phi2


def findPhisnoconst(chi,phiC):
    phi1spin,phi2spin=get_spins(chi,phiC)
    bounds = [(epsilon,phi1spin-epsilon), (phi2spin+epsilon,1-epsilon)]
    # bounds = [(epsilon,phiC-epsilon), (phiC+epsilon,1-epsilon)]
    initial_guess = (phi1spin*.9,phi2spin+epsilon)
    # initial_guess = (epsilon,1-epsilon)

    result = minimize(FH_Seperated, initial_guess, args=(chi,), method='Nelder-Mead', bounds=bounds)
    maxparams = result.x
    phi1,phi2 = min(maxparams), max(maxparams)
    return phi1,phi2,phi1spin,phi2spin
def getbinodal(chiC,phiC):
    bibin=[phiC]
    spinbin=[phiC]
    chibin =[1/chiC]
    chitest=chiC+resolution
    while chitest<(2):
        phil,phi2,s1,s2 = findPhisnoconst(chitest,phiC)
        bibin = np.concatenate(([phil], bibin, [phi2]))
        spinbin = np.concatenate(([s1],spinbin,[s2]))
        chibin = np.concatenate(([1/chitest], chibin, [1/chitest]))
        chitest+=resolution

    return bibin, spinbin, chibin

if __name__ =='__main__':
    phiC, chiC = get_critical_vals(N)
    print(phiC, chiC)
    bis,spins, chis = getbinodal(chiC, phiC)
    plt.plot(bis, chis, label='Binodal')
    plt.plot(spins, chis, label='Spinodal')
    plt.show()




