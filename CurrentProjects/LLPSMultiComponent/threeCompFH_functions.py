from threeCompFH_init import chi11,chi12,chi22, N1, N2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#########SOLVER CONSTS#############
res = .001
epsilon = 1e-15

def free_energy_3piece(phi1, phi2):
    if phi1>0 and phi2 > 0 and (phi1 + phi2 < 1):
        return (phi1/N1)*np.log(phi1) + (phi2/N2)*np.log(phi2) + (1-phi1-phi2)*np.log(1-phi1-phi2) - chi11*phi1**2 - chi22*phi2**2 - 2*chi12*phi1*phi2
    else:
        print("Unallowed phi regime \nphi1: ", phi1, "\nphi2: ", phi2)
        return 1e10
def fternary(phi1A, phi1B, phi2A, phi2B, vA, vB):
    phi1C, phi2C = 0,0 #######FIGURE OUT HOW TO DEFINE THIRD PHASE
    fT = vA* free_energy_3piece(phi1A,phi2A) + vB* free_energy_3piece(phi1B, phi2B) + (1 - vA - vB)*free_energy_3piece(phi1C,phi2C)
    ###########ALSO NEED VOLUME CONSTRAINTS################
    return fT

def fbinary(variables,bulk):
    phi1A, phi2A, v = variables
    phi1Bulk, phi2Bulk= bulk
    phi1B= (phi1Bulk- v*phi1A)/(1-v)
    phi2B= (phi2Bulk- v*phi2A)/(1-v)
    fB = v * free_energy_3piece(phi1A, phi2A) + (1 - v) * free_energy_3piece(phi1B, phi2B)
    return fB
def makeConstraints(bulk):
    bulk1, bulk2 = bulk
    def phiAConst(variables):
        return 1 - variables[0] - variables[1]    # This should be >= 0 for the constraint to be satisfied
    def phiBConst(variables):
        phi1B = (bulk1 - variables[2]*variables[0])/(1-variables[2])
        phi2B = (bulk2 - variables[2]*variables[1])/(1-variables[2])
        return 1 - phi1B - phi2B
    def maxV(variables):
        return 1 - variables[2]
    def minV(variables):
        return variables[2]
    def minphi1A(variables):
        return variables[0]
    def minphi2A(variables):
        return variables[1]
    def minphi1B(variables):
        phi1B = (bulk1 - variables[2]*variables[0])/(1-variables[2])
        return phi1B
    def minphi2B(variables):
        phi2B = (bulk2 - variables[2]*variables[1])/(1-variables[2])
        return phi2B
    def maxphi1A(variables):
        return 1- variables[0]
    def maxphi2A(variables):
        return 1- variables[1]
    def maxphi1B(variables):
        phi1B = (bulk1 - variables[2]*variables[0])/(1-variables[2])
        return 1- phi1B
    def maxphi2B(variables):
        phi2B = (bulk2 - variables[2]*variables[1])/(1-variables[2])
        return 1- phi2B

    return [{'type': 'ineq', 'fun': phiAConst}, {'type': 'ineq', 'fun': phiBConst}, {'type': 'ineq', 'fun': maxV}, {'type': 'ineq', 'fun': minV},{'type': 'ineq', 'fun': minphi1A},{'type': 'ineq', 'fun': minphi1B},{'type': 'ineq', 'fun': minphi2A},{'type': 'ineq', 'fun': minphi2B},{'type': 'ineq', 'fun': maxphi1A},{'type': 'ineq', 'fun': maxphi1B},{'type': 'ineq', 'fun': maxphi2A},{'type': 'ineq', 'fun': maxphi2B}]

def solveBinaryExample1(bulk1list, bulk2list):
    phiMinimizedLight = []
    phiMinimizedDense = []
    for bulk1 in bulk1list:
        for bulk2 in bulk2list:
            bulk= (bulk1,bulk2)
            initial_guess= [.05,.05,.6]
            bounds= [(epsilon, 1-epsilon),(epsilon, 1-epsilon),(epsilon, 1-epsilon)]
            const= makeConstraints(bulk)
            result = minimize(fbinary, initial_guess, args=(bulk,), method='trust-constr', bounds=bounds, constraints=const)
            minInput= result.x
            minP1A, minP2A, minV = minInput  ###########0: phi1A, 1: phi2A, 2: v
            print(minInput)
            if (fbinary((minP1A,minP2A,minV),bulk) < free_energy_3piece(bulk1,bulk2)):
                minP1B = (bulk1 - minV * minP1A) / (1 - minV)
                minP2B = (bulk2 - minV * minP2A) / (1 - minV)
                if minP1A>0and minP1A<1 and minP2A>0 and minP2A <1 and minV>0 and minV<1 and minP1B>0 and minP1B<1 and minP2B>0 and minP2B<1:
                    phiMinimizedLight.append((minP1A,minP2A))
                    phiMinimizedDense.append((minP1B,minP2B))
            else:pass
    return phiMinimizedDense,phiMinimizedLight

def pltScatter(pointsDense, pointsLight):
    phi1coordsA = [point[0] for point in pointsLight]
    phi2coordsA = [point[1] for point in pointsLight]
    phi1coordsB = [point[0] for point in pointsDense]
    phi2coordsB = [point[1] for point in pointsDense]

    plt.scatter(phi1coordsA,phi2coordsA, linewidths=0.05)
    plt.scatter(phi1coordsB, phi2coordsB, linewidths=0.05)
    plt.xlim((0,.25))
    plt.ylim((0,.25))

    plt.legend()

    plt.show()
    return



