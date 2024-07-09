from threeCompFH_init import chi11,chi12,chi22, N1, N2,checkTernary_Flag
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
def fternary(variables, bulk):
    bulk1,bulk2=bulk
    phi1A,phi1B,phi2A,phi2B,vA,vB = variables

    phi1C = (bulk1-vA*phi1A- vB*phi1B)/(1-vA-vB)
    phi2C = (bulk2-vA*phi2A-vB*phi2B)/(1-vA-vB)

    fT = vA * free_energy_inphase(phi1A, phi2A) + vB * free_energy_inphase(phi1B, phi2B) + (1 - vA - vB) * free_energy_inphase(
        phi1C, phi2C)
    ###########ALSO NEED VOLUME CONSTRAINTS################
    return fT
def checkEqualPotentialBI(a1,a2,b1,b2,v):
    thresh = .001
    jac1 = v*(df_dphi1(a1, a2) - df_dphi1(b1, b2))
    jac2 = v*(df_dphi2(a1, a2) - df_dphi2(b1, b2))
    jac3 = free_energy_inphase(a1,a2) - free_energy_inphase(b1,b2) + (b1 - a1) * df_dphi1(b1, b2) + (b2 - a2) * df_dphi2(
        b1, b2)
    if ((np.abs(jac1)<= thresh) and (np.abs(jac2) <= thresh) and (np.abs(jac3) <=thresh)):
        return True
    else: return False
def checkEqualPotentialTri(a1,a2,b1,b2,c1,c2, va,vb):
    # bulk1,bulk2=bulk
    # c1 = (bulk1 - va * a1 - vb * b1) / (1 - va - vb)
    # c2 = (bulk2 - va * a2 - vb * b2) / (1 - va - vb)
    thresh = .005

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
def getInitial(bulk):
    bulk1,bulk2 = bulk
    bounds = [(epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon)]

    initial= [1.0001*bulk1, .9999*bulk2, .3]#######EXAMPLE 3--- SEGREGATION ASYMMETRY-----NEGATIVE TIE LINES

    result = minimize(fbinary, initial, args=(bulk,), method='Nelder-Mead', bounds=bounds)
    inputs= result.x
    p1a,p2a,v = inputs[0],inputs[1],inputs[2]
    return p1a, p2a, v
def getInitialTer(bulk):
    bulk1,bulk2 = bulk
    bounds = [(epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon),(epsilon, 1 - epsilon),(epsilon, 1 - epsilon),(epsilon, 1 - epsilon)]
    #initial= [.999*bulk1, 1.01*bulk2,.999*bulk1, 1.01*bulk2,.1,.1]#######EXAMPLE 3--- SEGREGATION ASYMMETRY-----NEGATIVE TIE LINES
    initial= [.9*bulk1, 1.5*bulk2,1.5*bulk1, .9*bulk2,.15,.15]#######EXAMPLE 3--- SEGREGATION ASYMMETRY-----NEGATIVE TIE LINES

    #initial=[.025,.23,.025,.001,.1,.1]
    result = minimize(fternary, initial, args=(bulk,), method='Nelder-Mead', bounds=bounds)
    inputs= result.x
    p1a,p1b,p2a,p2b,va,vb = inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5]
    return p1a,p1b,p2a, p2b, va,vb
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
def verifyMinimizedTern(a1,b1,a2,b2,c1,c2,va,vb,bulk):
    bulk1,bulk2 = bulk
    phiMinimizedLight = []
    phiMinimizedDense = []
    phiTerA=[]
    phiTerB=[]
    phiTerC=[]


    ##GET A BINARY TO TEST:
    phi1ai, phi2ai, vi, = getInitial(bulk)
    # initial_guess= [.9*bulk1, .9*bulk2, .92]
    initial_guess = [phi1ai, phi2ai, vi]
    bounds = [(epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon)]
    const = makeConstraints(bulk)
    result = minimize(fbinary, initial_guess, args=(bulk,), method='SLSQP', bounds=bounds, constraints=const)
    minInput = result.x
    minP1A, minP2A, minV = minInput  ###########0: phi1A, 1: phi2A, 2: v

    ########IF TERNARY:###########
    if(fternary((a1,b1,a2,b2,va,vb),bulk)<free_energy_inphase(bulk1,bulk2) and fternary((a1,b1,a2,b2,va,vb),bulk)< fbinary((minP1A,minP2A,minV),bulk)):
        if a1>0 and a1<1 and b1>0 and b1<1 and a2>0 and a2<1 and b2>0 and b2<1 and c1>0 and c1<1 and c2>0 and c2<1 and (va+vb)>0 and (va + vb)<1 :
            if checkEqualPotentialTri(a1,a2,b1,b2,c1,c2,va,vb):
                if (calcDeterminant(a1,a2)>0 and calcDeterminant(b1,b2)>0 and calcDeterminant(c1,c2)>0):
                    if not phiTerA:
                        phiTerA.append((a1,a2))
                        phiTerB.append((b1,b2))
                        phiTerC.append((c1,c2))
    ########IF BINARY:############
    else:
        if (fbinary((minP1A, minP2A, minV), bulk) < free_energy_inphase(bulk1, bulk2)):
            minP1B = (bulk1 - minV * minP1A) / (1 - minV)
            minP2B = (bulk2 - minV * minP2A) / (1 - minV)
            if minP1A > 0 and minP1A < 1 and minP2A > 0 and minP2A < 1 and minV > 0 and minV < 1 and minP1B > 0 and minP1B < 1 and minP2B > 0 and minP2B < 1 :
                if (checkEqualPotentialBI(minP1A, minP2A, minP1B, minP2B, minV)):
                    if ((calcDeterminant(minP1A, minP2A) > 0) and (calcDeterminant(minP1B, minP2B) > 0)):
                        phiMinimizedLight.append((minP1A, minP2A))
                        phiMinimizedDense.append((minP1B, minP2B))

                    else:
                        pass
            else:
                pass
        else:
            pass
    if not phiMinimizedLight:
        return (phiTerA,phiTerB,phiTerC)
    else: return (phiMinimizedLight,phiMinimizedDense)
def solveBinaryExample1(bulkPairs):

    phiMinimizedLight = []
    phiMinimizedDense = []

    phiTriA = []
    phiTriB = []
    phiTriC = []

    for pair in bulkPairs:
        bulk1,bulk2 = pair
        print(pair)
        if bulk1+bulk2<1:
            if (calcDeterminant(bulk1,bulk2)<0):  ###########BULK IN CONCAVE/SPINODAL REGION################
                if checkTernary_Flag == 1:
                    bulk = (bulk1,bulk2)
                    phi1ai,phi1bi,phi2ai,phi2bi,vai,vbi = getInitialTer(bulk)
                    initial_guessTer=[phi1ai,phi1bi,phi2ai,phi2bi,vai,vbi]
                    #initial_guessTer=[.025,.23,.025,.001,.2,.2]
                    bounds = [(epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon), (epsilon, 1 - epsilon)]
                    const = makeConstraintsTer(bulk)
                    result = minimize(fternary,initial_guessTer,args=(bulk,),method='SLSQP',bounds=bounds,constraints=const)
                    minInputT = result.x
                    mtP1A,mtP1B,mtP2A,mtP2B,mtVA,mtVB = minInputT
                    mtP1C= (bulk1-mtVA*mtP1A-mtVB*mtP1B)/(1-mtVA-mtVB)
                    mtP2C= (bulk2-mtVA*mtP2A-mtVB*mtP2B)/(1-mtVA-mtVB)
                    ##########calced possible ternary phase################

                    phisMin = verifyMinimizedTern(mtP1A,mtP1B,mtP2A,mtP2B,mtP1C,mtP2C,mtVA,mtVB,bulk)
                    if len(phisMin)==3:
                        terMinA, terMinB, terMinC = phisMin

                        if terMinA and terMinB and terMinC:
                            if not phiTriA and not phiTriB and not phiTriC:
                                phiTriA.append(terMinA)
                                phiTriB.append(terMinB)
                                phiTriC.append(terMinC)
                        #########TERNARY
                    else:
                        biMinA, biMinB = phisMin
                        print(biMinA)
                        (phi1AMin,phi2AMin) = biMinA[0][0], biMinA[0][1]
                        (phi1BMin,phi2BMin) = biMinB[0][0],biMinB[0][1]
                        phiMinimizedLight.append((phi1AMin,phi2AMin))
                        phiMinimizedDense.append((phi1BMin,phi2BMin))
                else:
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
    return phiMinimizedDense,phiMinimizedLight,phiTriA,phiTriB,phiTriC
def pltScatter(pointsDense, pointsLight,tripoints,numSamples):

    triA,triB,triC = tripoints
    plt.figure(figsize=(5, 4), dpi=150)

    ##############PHASE BOUNDARY############################
    phi1coordsA = [point[0] for point in pointsLight]
    phi2coordsA = [point[1] for point in pointsLight]
    phi1coordsB = [point[0] for point in pointsDense]
    phi2coordsB = [point[1] for point in pointsDense]

    plt.scatter(phi1coordsA,phi2coordsA, s=1, c='b')
    plt.scatter(phi1coordsB, phi2coordsB, s=1, c='b')

    print('lights', pointsLight)
    print('denses',pointsDense)
    ###########TIELINES#######################
    # if numSamples< len(pointsLight):
    #     random_indices = random.sample(range(len(pointsLight)), numSamples)
    # else:
    #     numSamples = len(pointsLight)
    #     random_indices = random.sample(range(len(pointsLight)), numSamples)
    #
    # for i in random_indices:
    #     xA, yA = pointsLight[i]
    #     xB, yB = pointsDense[i]
    #     plt.plot([xA, xB], [yA, yB], linestyle=':', color='k', linewidth=0.5)  # Black dotted line

    #TerNARYTHIGNS
    treA = triA[0][0]
    treB = triB[0][0]
    treC = triC[0][0]
    print('triA',treA)
    print('triB',treB)
    print('triC',treC)

    triangle = plt.Polygon([treA, treB, treC], closed=True, fill=None, edgecolor='b', linewidth=.5)
    plt.gca().add_patch(triangle)

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



