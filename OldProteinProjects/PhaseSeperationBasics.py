import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import root

###constants##############
k = 1
TtoXconvConst = 1
NA = 1000
NB = 1

def criticalVals(NA,NB):
    phiC = (np.sqrt(NB) / (np.sqrt(NB) + np.sqrt(NA)))
    chiC = (1 + np.sqrt(NA)) / (2 * np.sqrt(NA))
    return phiC, chiC
def equilibrium_equation(chi, phi):
    return (1/NA)* phi * np.log(phi) + (1 - phi)*(1/NB) * np.log(1 - phi) + chi * phi * (1 - phi)
def equilibriumfirstderivative(chi,phi):
    return (1/NA) + np.log(phi)/NA - 1/NB - np.log(1-phi)/NB -2*chi*phi + chi
def equilibriumsecondderiative(chi,phi):
    return (1/(NA*phi))+ (1/(NB*(1-phi)))- 2*chi
def equilibriumthirdderiative(chi,phi):
    return (-1/(NA*phi**2))+ (1/(NB*(1-phi)**2))

def bimodaleqns(phis,chi):
    phi1,phi2 = phis
    # lighteqn = equilibriumfirstderivative(chi,phi1) - ((equilibrium_equation(chi,phi2)-equilibrium_equation(chi,phi1))/(phi2-phi1))
    # denseeqn = equilibriumfirstderivative(chi,phi2) - ((equilibrium_equation(chi,phi2)-equilibrium_equation(chi,phi1))/(phi2-phi1))
    lighteqn = equilibriumfirstderivative(chi, phi1)
    denseeqn = equilibriumfirstderivative(chi, phi2)
    return [lighteqn,denseeqn]

def getbinodal(chiC,phiC):
    phi1s = []
    phi2s = []
    chis = np.linspace(0.0001,chiC-.001,100)
    initial_guess = [0.01,.99]
    for chi in chis:
        result = fsolve(bimodaleqns, initial_guess, args=(chi,))
        phi1s.append(result[0])
        phi2s.append(result[1])
        print(chi, result)

    phi1s = np.sort(phi1s)
    phi2s = np.sort(phi2s)

    return phi1s, phi2s

phi_values = np.linspace(1.0e-10, .99999, 1000)
Ts = []
for i in phi_values:
    X = fsolve(equilibriumsecondderiative,.5,args=(i,))
    Ts.append(TtoXconvConst/(k*X))

phic, XC = criticalVals(NA,NB)
print(phic)
binodal1s, binodal2s = getbinodal(XC,phic)
chis = np.linspace(0.0001, XC , 100)

plt.plot(binodal1s, chis, label='Phi 1')
plt.plot(binodal2s, chis, label='Phi 2')
plt.show()


plt.figure()
plt.plot(phi_values,Ts)
plt.xlabel('Volume Fraction (phi)')
plt.ylabel('Temperature (T)')
plt.title(('T vs Volume Fraction Phase Diagram, NA =',NA,',NB = 1'))
plt.show()


