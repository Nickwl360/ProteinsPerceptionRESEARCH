import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize

###constants##############
epsilon = 1e-7
resolution = .001
NA = 1000
NB = 1

def criticalVals(NA,NB):
    phiC = (1 / (1 + np.sqrt(NA)))
    chiC = .5*(1 + 1/np.sqrt(NA))**2
    return phiC, chiC
def equilibriumsecondderiative(chi,phi):
    return (1 / (NA * phi)) + (1 / (NB * (1 - phi))) - 2 * chi
def Spinodaleqn(phi,chi):
    return np.sqrt(((1 / (NA * phi)) + (1 / (NB * (1 - phi))) - 2 * chi)**2)

def equilibrium_equation(chi, phi):
    return (1/NA)* phi * np.log(phi) + (1 - phi)*(1/NB) * np.log(1 - phi) + chi * phi * (1 - phi)
def FreeEnergyTotalnoconstraint(variables,chi):
    phi1,phi2 = variables
    phiC, XC = criticalVals(NA, NB)
    v = (phiC-phi2)/(phi1-phi2)
    eqn = v * equilibrium_equation(chi,phi1) + (1-v)*equilibrium_equation(chi,phi2)
    return eqn

def findSpinLow(chi,phiC):
    initial = phiC/2
    bounds = [(epsilon, phiC-epsilon)]
    result = minimize(Spinodaleqn, initial, args=(chi,),method='Powell',bounds=bounds)
    return result.x
def findSpinHigh(chi,phiC):
    initial = phiC+phiC/2
    bounds = [(phiC+epsilon, 1-epsilon)]
    result = minimize(Spinodaleqn, initial, args=(chi,),method='Nelder-Mead',bounds=bounds)
    return result.x


def findPhisnoconst(chi,phiC):
    phi1spin = findSpinLow(chi,phiC)[0]
    phi2spin= findSpinHigh(chi,phiC)[0]
    print(chi,phi1spin,phi2spin)
    bounds = [(epsilon,phi1spin-epsilon), (phi2spin+epsilon,1-epsilon)]
    # bounds = [(epsilon,phiC-epsilon), (phiC+epsilon,1-epsilon)]
    initial_guess = (phi1spin*.9,phi2spin+epsilon)
    # initial_guess = (epsilon,1-epsilon)

    maxL = minimize(FreeEnergyTotalnoconstraint, initial_guess, args=(chi,), method='Nelder-Mead', bounds=bounds)
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

phic, XC = criticalVals(NA,NB)
print(phic,XC)
phis,chis = getbinodal(XC,phic)
plt.plot(phis, chis, label='Binodal')

phi_values = np.linspace(1.0e-10, .99999, 1000)
Ts = []
for i in phi_values:
    X = fsolve(equilibriumsecondderiative,.5,args=(i,))
    Ts.append(1/X)

plt.plot(phi_values,Ts,label='Spinodal')
plt.legend()
plt.show()



