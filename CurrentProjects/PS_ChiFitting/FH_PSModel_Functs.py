import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from chi_fit_init import *
from scipy.optimize import brenth,root_scalar



def spin_yfromphi(phi,chi):
    phic = 1 / (1 + N ** (0.5))
    chic = 2*N / ((1 + N ** (0.5)) ** 2)
    tc = 1 / chic
    ti = tc

    d2f = lambda t: d2_FH(phi, t, chi)
    res = root_scalar(d2f, x0=ti, x1=ti*2, rtol=tol, bracket = (epsilon, 1e3))

    #print('phi,t:' ,phi, res.root, 'attempted')

    return -1*np.float64(res.root)
def get_critical_vals(N,T,chi):
    bounds = (epsilon,1-epsilon)
    res = minimize_scalar(spin_yfromphi,args=(chi,),method='bounded',bounds=bounds)
    (phic,tc) = (res.x, -1*res.fun)

    return phic, tc

def FH_free_energy(T, phi,chi):
    return (1/N)* phi * np.log(phi) + (1 - phi)* np.log(1 - phi) + chi/T * phi * (1 - phi)
def FH_Seperated(variables,T,phiC,chi):
    phi1,phi2 = variables
    v = (phiC-phi2)/(phi1-phi2)
    eqn = v * FH_free_energy(T, phi1,chi) + (1 - v) * FH_free_energy(T, phi2,chi)
    return eqn
def d2_FH(phi,T,chi):
    return (1 / (N * phi)) + (1 / (1 - phi)) - 2 * chi/T
def get_spins(T,phiC,chi):
    #phiMax = (1-2*phiS)/(1+qc)-epsilon
    phiMax = 1-epsilon
    phi1 = brenth(d2_FH, epsilon, phiC, args=(T,chi,))
    phi2 = brenth(d2_FH, phiC, phiMax,args = (T,chi,))
    return phi1,phi2
def findPhisnoconst(T,chi,phiC):
    phi1spin,phi2spin=get_spins(T,phiC,chi)
    bounds = [(epsilon,phi1spin - epsilon), (phi2spin+epsilon,1-epsilon)]
    # bounds = [(epsilon,phiC-epsilon), (phiC+epsilon,1-epsilon)]
    initial_guess = (phi1spin*.9,phi2spin+epsilon)
    # initial_guess = (epsilon,1-epsilon)

    result = minimize(FH_Seperated, initial_guess, args=(T,phiC,chi,), method='Nelder-Mead', bounds=bounds)
    maxparams = result.x
    phi1,phi2 = min(maxparams), max(maxparams)
    return phi1,phi2,phi1spin,phi2spin
def getbinodal(Tc,phiC):
    bibin=[phiC]
    spinbin=[phiC]
    Tbin =[Tc]
    print(Tc)
    Ttest=Tc - resolution
    while Ttest>(Tmin):

        phil,phi2,s1,s2 = findPhisnoconst(Ttest,phiC)
        bibin = np.concatenate(([phil], bibin, [phi2]))
        spinbin = np.concatenate(([s1],spinbin,[s2]))
        Tbin = np.concatenate(([Ttest], Tbin, [Ttest]))
        Ttest-=resolution

    return bibin, spinbin, Tbin

if __name__ =='__main__':

    phiC, tc = get_critical_vals(N,1)
    print(phiC, tc)
    bis,spins, chis = getbinodal(tc, phiC)
    print('Binodal for chi = ', chi)
    plt.plot(bis, chis, label='Binodal')
    plt.plot(spins, chis, label='Spinodal')
    plt.show()




