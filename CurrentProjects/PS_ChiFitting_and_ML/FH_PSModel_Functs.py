import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from CurrentProjects.PS_ChiFitting_and_ML.chi_fitting_from_data.chi_fit_init import epsilon,w3, omega3toggle,resolution,Tmin
from scipy.optimize import brenth,root_scalar

tol = 1e-4

def spin_yfromphi(phi,chi,N):
    phic = 1 / (1 + N ** (0.5))
    chic = 2*N / ((1 + N ** (0.5)) ** 2)
    tc = 1 / chic
    ti = tc

    d2f = lambda t: d2_FH(phi, t, chi,N)
    res = root_scalar(d2f, x0=ti, x1=ti*2, rtol=tol, bracket = (epsilon, 1e3))

    #print('phi,t:' ,phi, res.root, 'attempted')

    return -1*np.float64(res.root)
def get_critical_vals(chi,N):
    bounds = (epsilon,1-epsilon)
    res = minimize_scalar(spin_yfromphi,args=(chi,N),method='bounded',bounds=bounds)
    (phic,tc) = (res.x, -1*res.fun)

    return phic, tc

def FH_free_energy(T, phi,chi,N):
    f3b = 0
    if omega3toggle:
        f3b += (w3 - 1/6)*phi**3

    return (1/N)* phi * np.log(phi) + (1 - phi)* np.log(1 - phi) + chi/T * phi * (1 - phi) + f3b
def FH_Seperated(variables,T,phiC,chi,N):
    phi1,phi2 = variables
    v = (phiC-phi2)/(phi1-phi2)
    eqn = v * FH_free_energy(T, phi1,chi,N) + (1 - v) * FH_free_energy(T, phi2,chi,N)
    return eqn
def d2_FH(phi,T,chi,N):
    d2f3b = 0
    if omega3toggle:
        d2f3b+= 6*(w3 -1/6)*phi
    return (1 / (N * phi)) + (1 / (1 - phi)) - 2 * chi/T + d2f3b
def get_spins(T,phiC,chi,N):
    #phiMax = (1-2*phiS)/(1+qc)-epsilon
    phiMax = 1-epsilon
    phi1 = brenth(d2_FH, epsilon, phiC, args=(T,chi,N,))
    phi2 = brenth(d2_FH, phiC, phiMax,args = (T,chi,N,))
    return phi1,phi2
def findPhisnoconst(T,chi,phiC,N):
    phi1spin,phi2spin=get_spins(T,phiC,chi,N)
    bounds = [(epsilon,phi1spin - epsilon), (phi2spin+epsilon,1-epsilon)]
    # bounds = [(epsilon,phiC-epsilon), (phiC+epsilon,1-epsilon)]
    initial_guess = (phi1spin*.9-epsilon,phi2spin*1.1+epsilon)
    #initial_guess = (epsilon,1-epsilon)

    result = minimize(FH_Seperated, initial_guess, args=(T,phiC,chi,N,), method='Nelder-Mead', bounds=bounds)
    maxparams = result.x
    phi1,phi2 = min(maxparams), max(maxparams)
    return phi1,phi2,phi1spin,phi2spin
def getbinodal(Tc,phiC,chi,N):
    bibin=[phiC]
    spinbin=[phiC]
    Tbin =[Tc]
    print(Tc)
    Ttest=Tc - resolution
    while Ttest>(Tmin):

        phil,phi2,s1,s2 = findPhisnoconst(Ttest,chi,phiC,N)
        bibin = np.concatenate(([phil], bibin, [phi2]))
        spinbin = np.concatenate(([s1],spinbin,[s2]))
        Tbin = np.concatenate(([Ttest], Tbin, [Ttest]))
        Ttest-=resolution

    return bibin, spinbin, Tbin

if __name__ =='__main__':

    phiC, tc = get_critical_vals(chi=0.1,N=10)
    print(phiC, tc)
    bis,spins, chis = getbinodal(tc, phiC,N=10)
    plt.plot(bis, chis, label='Binodal')
    plt.plot(spins, chis, label='Spinodal')
    plt.show()




