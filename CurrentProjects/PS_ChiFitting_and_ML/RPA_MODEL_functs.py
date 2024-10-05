import numpy as np
from OldProteinProjects.SCDcalc import *
from scipy import integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.optimize import root_scalar
from scipy.optimize import brenth, brent
import math
import time

########################ConstANTS################################
T0=1e8
iterlim=250
MINMAX=25
thresh = 1e-12
phiS = 0.0
epsilon = 1e-18


###############STRUCTURE FACTORS#############################
def xee(k,protein):

    k2 = k * k
    exp_factor = -1 / 6 *  k2
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.xeeSig[:protein.N] * exp_vals)
    return gksum / protein.N
def gk(k,protein):
    k2 = k * k
    exp_factor = -1 / 6  * k2
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.gkSig[:protein.N] * exp_vals)
    return gksum / protein.N
def ck(k,protein):
    k2 = k * k
    exp_factor = -1 / 6 *  k2
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.ckSig[:protein.N] * exp_vals)
    return gksum / protein.N

def d1_xee(k,protein):
    k2 = k * k
    exp_factor = -1 / 6  * k2
    coeff = -k2 / 6
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.xeeSig[:protein.N ] * protein.i_vals * exp_vals * coeff)
    return gksum / protein.N
    # return k*k*k*k/36*np.mean(sigs*L2*L2*np.exp((-1/6)*x*k*k*L))
    #return -k * k / 6 * np.sum(sigS * L * np.exp((-1 / 6) * x * k * k * L))/N
def d1_gk(k,protein):
    k2 = k * k
    exp_factor = -1 / 6  * k2
    coeff = -k2 / 6
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.gkSig[:protein.N ] * protein.i_vals * exp_vals * coeff)
    return gksum / protein.N
def d1_ck(k,protein):
    k2 = k * k
    exp_factor = -1 / 6  * k2
    coeff = - k2 / 6
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.ckSig[:protein.N] * protein.i_vals * exp_vals * coeff)
    return gksum / protein.N

def d2_xee(k,protein):

    k2 = k * k
    exp_factor = -1 / 6  * k2
    coeff = k2 * k2 / 36
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.xeeSig[:protein.N]* exp_vals * i2 * coeff)
    return gksum / protein.N
def d2_gk(k,protein):
    # return k*k*k*k/36*np.mean(sigs*L2*L2*np.exp((-1/6)*x*k*k*L))
    k2 = k * k
    exp_factor = -1 / 6  * k2
    coeff = k2 * k2 / 36
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.gkSig[:protein.N] * exp_vals * i2 * coeff)
    return gksum / protein.N
def d2_ck(k,protein):
    # return k*k*k*k/36*np.mean(sigs*L2*L2*np.exp((-1/6)*x*k*k*L))
    k2 = k * k
    exp_factor = -1 / 6  * k2
    coeff = k2 * k2 / 36
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.ckSig[:protein.N ] * exp_vals * i2 * coeff)
    return gksum / protein.N


#################FREE ENERGIES#####################################
def ftot_rg(phiM, Y, protein):
    if phiM > 1 or phiM < 0 or phiS > 1 or phiS < 0:
        print('illegal phi range detected')
        return np.nan

    ftot = entropy(phiM,protein) + fion(phiM, Y,protein ) + rgFP(phiM, Y,protein)
    #this is where we add lili's predicted omega2 values.
    ftot += protein.w2*phiM*phiM/2

    #2*np.pi*phiM*phiM/3  ##f0 term old version

    if protein.W3_TOGGLE==1:
        ftot += (protein.w3 - 1/6)*phiM**3

    return ftot
def rgFPint(k,Y,phiM,protein):
    xe = xee(k, protein)
    g = gk(k, protein)
    c = ck(k, protein)
    v = protein.w2*np.exp(-1*k*k/6)
    nu = k*k*Y/4/np.pi + protein.qc*phiM + phiS
    N1 = nu + phiM*(g*nu*v + xe) +v*phiM*phiM*(g*xe-c*c)
    A = N1/nu
    B= 1 + protein.Q/nu
    return (k*k/4/np.pi/np.pi)*np.log(A/B)
def rgFP(phiM, Y,protein):

    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(rgFPint, lowerlim, upperlim, args=(Y, phiM,protein,), limit=iterlim)
    fp = result[0]
    return fp
def fion(phiM, Y,protein):
    kl = np.sqrt(4 * np.pi * (phiS + protein.qc * phiM) / Y)
    return (-1 / (4 * np.pi)) * (np.log(1 + kl) - kl + .5 * kl * kl)
def s_1comp(x):
    ###THIS IS FROM LIN TO SPEED UP AND AVOID ERRORS
    return (x > epsilon)*x*np.log(x + (x < epsilon)) + 1e5*(x<0)
def entropy(phiM,protein):
    phiC = protein.qc * phiM
    phiW = 1 - phiM - phiS - phiC
    #################FIGURE OUT LOGIC FOR 0s
    return s_1comp(phiM)/protein.N + s_1comp(phiS)+ s_1comp(phiC) + s_1comp(phiW)

################FIRST DERIVATIVES##############################################################
def ds_1comp(x):
    ##ALSO FRM LIN FOR SPEED AND NO ERRORS
    return (np.log(x + 1e5 * (x <= 0)) + 1) * (x > 0)
def dfpintegrand(k,Y,phiM,protein):
    xe = xee(k,protein)
    g = gk(k,protein)
    c = ck(k,protein)
    ionConst = k*k*Y/(4*np.pi) + phiS + protein.qc*phiM
    v2 = protein.w2*np.exp(-k*k/6)
    c2 = c*c

    num = -2*c2*v2*phiM + g*v2*ionConst + 2*xe*g*v2*phiM + xe
    den = phiM*(c2*(-1*v2)*xe + xe*g*v2 +xe) + g*ionConst*v2 + ionConst

    return k*k*num/den
def d1_Frg_dphi(phiM,Y,protein):
    phic = protein.qc*phiM
    phiW = 1 - phiM - phic - phiS

    ###d1 entropy
    ds_dphi = (ds_1comp(phiM)/protein.N + protein.qc*ds_1comp(phic) - (1+protein.qc)*ds_1comp(phiW))*(phiM>0)

    ##d1 screening
    c = 4*np.pi/Y
    rho = protein.qc*phiM + phiS
    k = np.sqrt(c*rho)

    temp = -k/2/(1+k)*(1/Y)
    dfion_dphi=temp*protein.qc*(phiM>0)
    #dfion_dphi= (-1*np.sqrt(np.pi)*qc*np.sqrt((phic + phiS)/Y))/(Y*(2*np.sqrt(np.pi)*np.sqrt((phic)/Y)+1))

    #d1 fprotein
    upper,lower = 0,np.inf
    result = integrate.quad(dfpintegrand, lower, upper, args=(Y,phiM,protein), limit=iterlim)
    dfp_dphi = result[0] / (4 * np.pi * np.pi)

    d1_ftot = ds_dphi + dfion_dphi + dfp_dphi
    # w2 part from lili
    d1_ftot += protein.w2*phiM #d1f0

    if protein.W3_TOGGLE ==1:
        d1_ftot += 3*(protein.w3 - 1/6)*phiM**2

    return d1_ftot

#####################SECOND DERIVATIVE RG FREE ENERGIES############################################################
def d2s_1comp(x):
    return (x>0)/(x + (x==0))
def d2_FP_toint(k, Y, phiM,protein):
    phic = protein.qc*phiM
    k2 = k*k
    rho = k2 * Y / (4 * np.pi) + phiS + phic
    qc2, phi2, rho2 =  protein.qc * protein.qc, phiM * phiM, rho * rho
    xe = xee(k, protein)
    g = gk(k,protein)
    c = ck(k, protein)

    v2 = protein.w2 * np.exp(-1 * k2 / 6)
    D = xe * g - c * c
    vp2 = v2 * phi2*D / rho
    vp1= 2*v2*phiM*D/rho
    vr = xe/rho + v2*g

    num1 = 2*v2*D
    den = vp2 + phiM*vr + 1

    num2 = (vp1 + vr)*(vp1 + vr)
    #not sure if k2 should be here or not
    return k2*(num1/(rho*den) - num2/(den*den))

def d2_Frg_phiM(phiM,Y,protein):
    qc = protein.qc
    phic = qc * phiM
    phiW = 1 - phiM - phic - phiS

    #################Entropyd2##########
    d2s = (d2s_1comp(phiM) / protein.N + qc * qc * d2s_1comp(phic) + (1 + qc) * (1 + qc) * d2s_1comp(phiW)) * (phiM > 0)

    #####d2Fion#################
    c = 4 * np.pi / Y
    rho = phic + phiS
    k = np.sqrt((c) * (rho))
    ##THIS IS FROM LIN
    tp = qc / (1 + k) * (phiM > 0)
    temp = -1 * np.pi * (1 / Y) * (1 / Y) / (k + (k == 0)) * (k > 0)
    d2fion = temp * tp * tp

    # lili ML value
    d2f0 = protein.w2

    #################Electrofreeenergyd2###########
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(d2_FP_toint, lowerlim, upperlim, args=(Y, phiM,protein), limit=iterlim)
    d2fp = result[0] / (4 * np.pi * np.pi)

    d2_ftot =  np.float64((d2s + d2fp + d2fion + d2f0))


    if protein.W3_TOGGLE == 1:
        d2_ftot += 6*(protein.w3 - 1/6)*phiM

    return d2_ftot

###############################################################################################################
#######SOLVER METHODS BELOW####################################################################################
###############################################################################################################

def spin_yfromphi(phiM,protein):
    N = protein.N
    pc = 1 / (1 + N ** (0.5))
    tc = 2*N / ((1 + N ** (0.5)) ** 2)
    uc = 1 / tc
    # initial values
    ui = uc

    d2f = lambda u: d2_Frg_phiM(phiM, 1/u, protein)
    try:
        ures = root_scalar(d2f,x0=ui,x1=ui/2, rtol=thresh, bracket = (1e-4, 1e3))
        if ures.converged:
            return np.float64(ures.root)
    except (ValueError,RuntimeError) as e:
        return np.nan

    # print('phi,t:' ,phiM, ures.root, 'attempted')
    # print(d2_Frg_phiM(phiM,ures.root,phiS),'this is d2')
    #return np.float64(ures.root)
def findCrit(protein):
    bounds = (epsilon,1-epsilon)

    result = minimize_scalar(spin_yfromphi,args=(protein),method='bounded', bounds=bounds)
    (pf,uf) = (result.x, result.fun)
    tf = 1/uf
    return np.float64(pf), np.float64(tf)
def findSpins(Y,protein):
    ##THIS FROM LIN
    phiMax = (1-2*phiS)/(1+protein.qc)-epsilon
    phi1 = brenth(d2_Frg_phiM, epsilon, protein.phiC, args=(Y,protein))
    phi2 = brenth(d2_Frg_phiM, protein.phiC, phiMax,args = (Y,protein))
    return phi1,phi2

def FBINODAL(variables,Y,phiBulk,protein):

    phi1,phi2 = variables
    if math.isnan(phi1) or math.isnan(phi2): return 1e20
    v = (phi2-phiBulk)/(phi2-phi1)
    eqn = v * ftot_rg(phi1, Y,protein) + (1 - v) * ftot_rg(phi2, Y, protein)

    ftot = T0* (eqn - ftot_rg(phiBulk, Y, protein))

    return ftot
    #return np.abs(ftot)#_differentiable
def Jac_rgRPA(vars,Y,phiB,protein):
    phi1=vars[0]
    phi2=vars[1]

    if math.isnan(phi1) or math.isnan(phi2):
        print('Phis are Nan')
        return np.empty(2)

    v = (phi2-phiB)/(phi2-phi1)

    f1 = ftot_rg(phi1, Y,protein)
    f2 = ftot_rg(phi2, Y,protein)
    df1 = d1_Frg_dphi(phi1, Y, protein)
    df2 = d1_Frg_dphi(phi2, Y, protein)

    J = np.empty(2)
    J[0] = v*( (f1-f2)/(phi2-phi1) + df1)
    J[1] = (1-v)*( (f1-f2)/(phi2-phi1) + df2)

    return J*T0

def minFtotal(Y,protein):

    phi1spin,phi2spin = findSpins(Y,protein)

    print(phi1spin, phi2spin, 'SPINS LEFT/RIGHT')

    #phiB = (phi1spin+phi2spin)/2
    phiB = protein.phiC

    assert np.isfinite(phi1spin), "phi1spin is not a finite number"
    assert np.isfinite(phi2spin), "phi2spin is not a finite number"

    ### GET CONSTRAINTS ###
    phiMax = (1-2*phiS)/(1 + protein.qc) - epsilon ### FROM LIN ###
    #bounds = [(phi1spin/10, phi1spin - epsilon), (phi2spin + epsilon, 1-epsilon) ]
    bounds = [(phi1spin/10, phi1spin - epsilon), (phi2spin+epsilon, phi2spin*1.6*(protein.Yc/Y)**2) ]

    t0 = time.time()

    ### MINIMIZER ### DEPENDS ON IF STARTING AT TOP OR NOT ###
    if Y!=protein.Yc:
        ### METHODS TO CHOOSE ### SCIPY.OPTIMIZE.MINIMIZE ###
        M1 = 'TNC'
        M2 = 'L-BFGS-B'

        ### MAKE INITIAL ### IN PROGRESS ###
        #initial_guess=(np.float64(phi1spin*0.9),np.float64(phi2spin*1.08))
        initial_guess=(np.float64(phi1spin*.8 - phi1spin*1.75*(protein.Yc/Y - 1)),np.float64(phi2spin*1.25 + phi2spin*2.25*((protein.Yc/Y) - 1)))

        ### DEFINE SPINS AND MINIMIZE ### USES MULTIPLE MINIMIZERS AND CHECKS VALIDITY ###
        maxL1 = minimize(FBINODAL, initial_guess, args=(Y, phiB,protein), method=M1, jac=Jac_rgRPA, bounds=bounds, options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20})# , 'maxfun':MINMAX})
        maxL2 = minimize(FBINODAL, initial_guess, args=(Y, phiB,protein), method=M2, jac=Jac_rgRPA, bounds=bounds, options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20})#, 'maxfun':MINMAX})
        #print(M2, 'M2 minimized, beginning,', M3 ,'M3\n')

        maxL3 = maxL2

        if maxL1 == None: maxL1=maxL2
        elif maxL2 ==None: maxL2 =maxL1

        ### FINDING LOWEST SOLUTION ###
        if(maxL1.fun<=maxL2.fun and maxL1.fun<=maxL3.fun):
            phi1min = min(maxL1.x)
            phi2min = max(maxL1.x)
        elif(maxL2.fun<= maxL1.fun and maxL2.fun<=maxL3.fun):
            phi1min = min(maxL2.x)
            phi2min = max(maxL2.x)
        else:
            phi1min = min(maxL3.x)
            phi2min = max(maxL3.x)
    else:
        ### TOP OF BINODAL GRAPH ### THIS ALWAYS WORKS ###
        initial_guess = (np.float64(phi1spin * .9), np.float64(phi2spin * 1.1))
        spins= [phi1spin,phi2spin]
        maxL1 = minimize(FBINODAL, initial_guess, args=(Y, phiB, spins), method='TNC', jac=Jac_rgRPA, bounds=bounds)#, options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20 })#, 'maxfun':MINMAX})
        phi1min = min(maxL1.x)
        phi2min = max(maxL1.x)

    ### PRINTING RESULTS FROM MINIMIZER ###
    v = (phi2min-phiB)/(phi2min-phi1min)

    print('\nMINIMIZER COMPLETE FOR Y = ',Y, 'MIN VALUES: phi1,phi2, v = ',phi1min,phi2min,v)
    print('\nThis step took ', time.time()-t0, 's')
    return phi1spin, phi2spin, phi1min,phi2min
def getBinodal(protein):

    biphibin= np.array([protein.phiC])
    sphibin = np.array([protein.phiC])
    Ybin = np.array([protein.Yc])

    for Ytest in protein.Yspace:

        spin1,spin2, phi1,phi2 = minFtotal(Ytest, protein)
        biphibin, sphibin = biphibin.flatten(), sphibin.flatten()

        #if phi1<phiLlast and phi2>phiDlast:
        if True:
            phi1=np.array([phi1])
            phi2=np.array([phi2])
            spin1 = np.array([spin1])
            spin2 = np.array([spin2])
            biphibin = np.concatenate((phi1, biphibin, phi2))
            sphibin = np.concatenate((spin1, sphibin, spin2))
            Ybin = np.concatenate(([Ytest], Ybin, [Ytest]))

    return sphibin, biphibin, Ybin
