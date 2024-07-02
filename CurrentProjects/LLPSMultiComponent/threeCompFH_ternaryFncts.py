from threeCompFH_init import chi11,chi12,chi22, N1, N2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
from threeCompFH_functions import *

#########SOLVER CONSTS#############
epsilon = 1e-15

def fternary(phi1A, phi1B, phi2A, phi2B, vA, vB, bulk):
    bulk1,bulk2=bulk
    phi1C = (bulk1-vA*phi1A- vB*phi1B)/(1-vA-vB)
    phi2C = (bulk2-vA*phi2A-vB*phi2B)/(1-vA-vB)

    fT = vA * free_energy_inphase(phi1A, phi2A) + vB * free_energy_inphase(phi1B, phi2B) + (1 - vA - vB) * free_energy_inphase(
        phi1C, phi2C)
    ###########ALSO NEED VOLUME CONSTRAINTS################
    return fT

def getInitialTer(bulk):
    bulk1,bulk2 = bulk
    bounds = [(epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon),(epsilon, 1 - epsilon),(epsilon, 1 - epsilon),(epsilon, 1 - epsilon)]
    initial= [1.0001*bulk1, .9999*bulk2,1.0001*bulk1, .9999*bulk2,.3,.3]#######EXAMPLE 3--- SEGREGATION ASYMMETRY-----NEGATIVE TIE LINES
    result = minimize(fternary, initial, args=(bulk,), method='Nelder-Mead', bounds=bounds)
    inputs= result.x
    p1a,p1b,p2a,p2b,va,vb = inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5]
    return p1a,p1b,p2a, p2b, va,vb

def makeConstraintsTer(bulk):
    bulk1, bulk2 = bulk       #0,1,2,3,4,5= 1a,1b, 2a,2b,va,vb
    def phiAConst(variables):
        return 1 - variables[0] - variables[2]    # This should be >= 0 for the constraint to be satisfied
    def phiBConst(variables):
        return 1 - variables[1] - variables[3]
    def phiCConst(variables):
        phi1C = (bulk1 - variables[4]*variables[0] - variables[5]*variables[1])/(1-variables[4]-variables[5])
        phi2C = (bulk2 - variables[4]*variables[2] - variables[5]*variables[3])/(1-variables[4]-variables[5])
        return 1 - phi1C - phi2C
    def maxVa(variables):
        return 1 - variables[4]
    def minVa(variables):
        return variables[4]
    def maxVb(variables):
        return 1- variables[5]
    def minVb(variables):
        return variables[5]
    def maxVc(variables):
        vc = 1- variables[4]-variables[5]
        return 1-vc
    def minVc(variables):
        vc = 1- variables[4]-variables[5]
        return vc
    def minphi1A(variables):
        return variables[0]
    def minphi2A(variables):
        return variables[1]
    def minphi1B(variables):
        return variables[2]
    def minphi2B(variables):
        return variables[3]
    def minphi1C(variables):
        phi1C = (bulk1 - variables[4]*variables[0] - variables[5]*variables[1])/(1-variables[4]-variables[5])
        return phi1C
    def minphi2C(variables):
        phi2C = (bulk2 - variables[4]*variables[2] - variables[5]*variables[3])/(1-variables[4]-variables[5])
        return phi2C
    def maxphi1A(variables):
        return 1- variables[0]
    def maxphi2A(variables):
        return 1- variables[1]
    def maxphi1B(variables):
        return 1- variables[2]
    def maxphi2B(variables):
        return 1- variables[3]
    def maxphi1C(variables):
        phi1C = (bulk1 - variables[4]*variables[0] - variables[5]*variables[1])/(1-variables[4]-variables[5])
        return 1- phi1C
    def maxphi2C(variables):
        phi2C = (bulk2 - variables[4]*variables[2] - variables[5]*variables[3])/(1-variables[4]-variables[5])
        return 1- phi2C

    return [{'type': 'ineq', 'fun': phiAConst}, {'type': 'ineq', 'fun': phiBConst},{'type': 'ineq', 'fun': phiCConst}, {'type': 'ineq', 'fun': maxVa},{'type': 'ineq', 'fun': maxVb}, {'type': 'ineq', 'fun': minVa}, {'type': 'ineq', 'fun': minVb}, {'type': 'ineq', 'fun': minVc},{'type': 'ineq', 'fun': maxVc},{'type': 'ineq', 'fun': minphi1A},{'type': 'ineq', 'fun': minphi1B},{'type': 'ineq', 'fun': minphi2A},{'type': 'ineq', 'fun': minphi2B},{'type': 'ineq', 'fun': minphi1C},{'type': 'ineq', 'fun': minphi2C},{'type': 'ineq', 'fun': maxphi1A},{'type': 'ineq', 'fun': maxphi1B},{'type': 'ineq', 'fun': maxphi2A},{'type': 'ineq', 'fun': maxphi2B},{'type': 'ineq', 'fun': maxphi1C},{'type': 'ineq', 'fun': maxphi2C}]

def checkEqualPotentialTri(a1,a2,b1,b2,c1,c2, va,vb):
    # bulk1,bulk2=bulk
    # c1 = (bulk1 - va * a1 - vb * b1) / (1 - va - vb)
    # c2 = (bulk2 - va * a2 - vb * b2) / (1 - va - vb)
    thresh = .01

    dfT_da1 = va*(df_dphi1(a1,a2)-df_dphi1(c1,c2))
    dfT_da2 = va*(df_dphi2(a1,a2)-df_dphi2(c1,c2))
    dfT_db1 = vb*(df_dphi1(b1,b2)-df_dphi1(c1,c2))
    dfT_db2 = vb*(df_dphi2(b1,b2)-df_dphi2(c1,c2))
    dfT_dva = free_energy_inphase(a1,a2)-free_energy_inphase(c1,c2) + df_dphi1(c1,c2)*(c1-a1) + df_dphi2(c1,c2)*(c2-a2)
    dfT_dvb = free_energy_inphase(b1,b2)-free_energy_inphase(c1,c2) + df_dphi1(c1,c2)*(c1-b1) + df_dphi2(c1,c2)*(c2-b2)

    if ((np.abs(dfT_da1)<= thresh) and (np.abs(dfT_da2) <= thresh) and (np.abs(dfT_db1) <=thresh) and (np.abs(dfT_db2 <=thresh))and (np.abs(dfT_dva) <=thresh)and (np.abs(dfT_dvb) <=thresh)):
        return True
    else: return False

def calcDeterminant(phi1,phi2):
    d2f_d21 = (1/(N1*phi1)) + (1/(1-phi1-phi2)) - 2*chi11
    d2f_d22 = (1/(N2*phi2)) + (1/(1-phi1-phi2)) - 2*chi22
    d2f_d1d2 = (1/(1-phi1-phi2)) - 2 * chi12
    return (d2f_d21*d2f_d22 - d2f_d1d2*d2f_d1d2)

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



