from threeCompFH_init import chi11,chi12,chi22, N1, N2,checkTernary_Flag
from threeCompFH_ternaryFncts import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random

#########SOLVER CONSTS#############
epsilon = 1e-15

def free_energy_inphase(phi1, phi2):
    if phi1>0 and phi2 > 0 and (phi1 + phi2 < 1):
        return (phi1/N1)*np.log(phi1) + (phi2/N2)*np.log(phi2) + (1-phi1-phi2)*np.log(1-phi1-phi2) - chi11*phi1**2 - chi22*phi2**2 - 2*chi12*phi1*phi2
    else:
        #print("Unallowed phi regime \nphi1: ", phi1, "\nphi2: ", phi2)
        return 1e12

def df_dphi1(phi1,phi2):
    eqn = (np.log(phi1)/N1) + (1/N1) - np.log(1-phi1-phi2)- 1 - 2*chi11*phi1 - 2*chi12*phi2
    return eqn
def df_dphi2(phi1,phi2):
    eqn = (np.log(phi2)/N2) + (1/N2) - np.log(1-phi1-phi2)- 1 - 2*chi22*phi2 - 2*chi12*phi1
    return eqn
def fbinary(variables,bulk):
    phi1A, phi2A, v = variables
    phi1Bulk, phi2Bulk= bulk
    phi1B= (phi1Bulk- v*phi1A)/(1-v)
    phi2B= (phi2Bulk- v*phi2A)/(1-v)
    fB = v * free_energy_inphase(phi1A, phi2A) + (1 - v) * free_energy_inphase(phi1B, phi2B)
    return fB
def checkEqualPotentialBI(a1,a2,b1,b2,v):
    thresh = .001
    jac1 = v*(df_dphi1(a1, a2) - df_dphi1(b1, b2))
    jac2 = v*(df_dphi2(a1, a2) - df_dphi2(b1, b2))
    jac3 = free_energy_inphase(a1,a2) - free_energy_inphase(b1,b2) + (b1 - a1) * df_dphi1(b1, b2) + (b2 - a2) * df_dphi2(
        b1, b2)
    if ((np.abs(jac1)<= thresh) and (np.abs(jac2) <= thresh) and (np.abs(jac3) <=thresh)):
        return True
    else: return False
def calcDeterminant(phi1,phi2):
    d2f_d21 = (1/(N1*phi1)) + (1/(1-phi1-phi2)) - 2*chi11
    d2f_d22 = (1/(N2*phi2)) + (1/(1-phi1-phi2)) - 2*chi22
    d2f_d1d2 = (1/(1-phi1-phi2)) - 2 * chi12
    return (d2f_d21*d2f_d22 - d2f_d1d2*d2f_d1d2)
def getInitial(bulk):
    bulk1,bulk2 = bulk
    bounds = [(epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon)]

    initial= [1.0001*bulk1, .9999*bulk2, .3]#######EXAMPLE 3--- SEGREGATION ASYMMETRY-----NEGATIVE TIE LINES

    result = minimize(fbinary, initial, args=(bulk,), method='Nelder-Mead', bounds=bounds)
    inputs= result.x
    p1a,p2a,v = inputs[0],inputs[1],inputs[2]
    return p1a, p2a, v
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
def solveBinaryExample1(bulkPairs):
    phiMinimizedLight = []
    phiMinimizedDense = []

    for pair in bulkPairs:
        bulk1,bulk2 = pair

        if bulk1+bulk2<1:
            if (calcDeterminant(bulk1,bulk2)<0):  ###########BULK IN CONCAVE/SPINODAL REGION################
                if checkTernary_Flag == 1:
                    phi1ai,phi1bi,phi2ai,phi2bi,vai,vbi = getInitialTer(bulk)
                    initial_guessTer=[phi1ai,phi1bi,phi2ai,phi2bi,vai,vbi]
                    bounds = [(epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon)]
                    const = makeConstraintsTer(bulk)
                    result = minimize(fternary,initial_guessTer,args=(bulk,),method='SLSQP',bounds=bounds,constraints=const)
                    minInputT = result.x
                    mtP1A,mtP1B,mtP2A,mtP2B,mtVA,mtVB = minInputT
                    mtP1C= (bulk1-mtVA*mtP1A-mtVB*mtP1B)/(1-mtVA-mtVB)
                    mtP2C= (bulk2-mtVA*mtP2A-mtVB*mtP2B)/(1-mtVA-mtVB)

                bulk= (bulk1,bulk2)
                #initial_guess= [.05,.05,.5]
                phi1ai, phi2ai, vi, = getInitial(bulk)
                #initial_guess= [.9*bulk1, .9*bulk2, .92]
                initial_guess=[phi1ai,phi2ai,vi]
                bounds= [(epsilon, 1-epsilon),(epsilon, 1-epsilon),(epsilon, 1-epsilon)]
                const= makeConstraints(bulk)
                result = minimize(fbinary, initial_guess, args=(bulk,), method='SLSQP', bounds=bounds, constraints=const)
                minInput= result.x
                minP1A, minP2A, minV = minInput  ###########0: phi1A, 1: phi2A, 2: v

                print(bulk, '\n',minInput)

                if (fbinary((minP1A,minP2A,minV),bulk) < free_energy_inphase(bulk1, bulk2)):
                    minP1B = (bulk1 - minV * minP1A) / (1 - minV)
                    minP2B = (bulk2 - minV * minP2A) / (1 - minV)
                    if minP1A>0 and minP1A<1 and minP2A>0 and minP2A <1 and minV>0 and minV<1 and minP1B>0 and minP1B<1 and minP2B>0 and minP2B<1:
                        if(checkEqualPotentialBI(minP1A, minP2A, minP1B, minP2B, minV)):
                            if((calcDeterminant(minP1A,minP2A)>0) and (calcDeterminant(minP1B,minP2B)>0)):
                                phiMinimizedLight.append((minP1A,minP2A))
                                phiMinimizedDense.append((minP1B,minP2B))

                            else:pass
                    else:pass
                else:pass
            else:pass
        else:pass
    return phiMinimizedDense,phiMinimizedLight

def pltScatter(pointsDense, pointsLight, numSamples):

    plt.figure(figsize=(5, 4), dpi=150)

    ##############PHASE BOUNDARY############################
    phi1coordsA = [point[0] for point in pointsLight]
    phi2coordsA = [point[1] for point in pointsLight]
    phi1coordsB = [point[0] for point in pointsDense]
    phi2coordsB = [point[1] for point in pointsDense]

    plt.scatter(phi1coordsA,phi2coordsA, s=1, c='b')
    plt.scatter(phi1coordsB, phi2coordsB, s=1, c='b')

    ###########TIELINES#######################
    random_indices = random.sample(range(len(pointsLight)), numSamples)
    for i in random_indices:
        xA, yA = pointsLight[i]
        xB, yB = pointsDense[i]
        plt.plot([xA, xB], [yA, yB], linestyle=':', color='k', linewidth=0.5)  # Black dotted line

    #################BOUNDARY AND OTHER SETTINGS###################
    plt.plot([1, 0], [0, 1], linestyle='-', color='r', linewidth=1, label='Boundary Line')  # Solid black line

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)  # Opaque grid lines

    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('phi1')
    plt.ylabel('phi2')
    #plt.title(('FH Phase Diagram: X11= ',chi11,'X22= ',chi22,'X12= ',chi12))

    plt.show()
    return


