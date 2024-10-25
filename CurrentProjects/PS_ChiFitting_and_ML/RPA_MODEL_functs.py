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
import warnings
from scipy.integrate import IntegrationWarning
from scipy.optimize import OptimizeWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=IntegrationWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

########################ConstANTS################################
T0=500
iterlim=500
epsabs = 1e-12
epsrel = 1e-12
MINMAX=25
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
    if phiM > 1 or phiM < 0 or protein.phiS > 1 or protein.phiS < 0:
        print('illegal phi range detected')
        return np.nan

    ftot = entropy(phiM,protein) + fion(phiM, Y,protein ) + rgFP(phiM, Y,protein)
    #this is where we add lili's predicted omega2 values.
    ftot += protein.w2*phiM*phiM/2

    #testing old version
    #ftot+= 2*np.pi*phiM*phiM/3

    if protein.W3_TOGGLE==1:
        ftot += (protein.w3 - 1/6)*phiM**3
    if protein.crowding_toggle == 1:
        ftot += -1*protein.epsC*phiM*phiM

    return ftot
def rgFPint(k,Y,phiM,protein):
    xe = xee(k, protein)
    g = gk(k, protein)
    c = ck(k, protein)
    v2 = protein.w2 * np.exp(-1 * k * k / 6)
    #testing fp with old version
    #v2 = 4 * np.pi / 3 * np.exp(-1 * k * k / 6) #fails to find crit

    rho = k * k * Y / 4 / np.pi + protein.qc * phiM + protein.phiS*2

    a = phiM*(xe / rho + v2 * g)
    b = (phiM * phiM * v2 / rho) * (g * xe - c * c)
    return (k*k/4/np.pi/np.pi)*np.log(1+a+b)
def rgFP(phiM, Y,protein):

    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(rgFPint, lowerlim, upperlim, args=(Y, phiM,protein,), limit=iterlim, epsabs=epsabs,epsrel=epsrel)
    fp = result[0]
    return fp
def fion(phiM, Y,protein):
    kl = np.sqrt(4 * np.pi * (protein.phiS*2 + protein.qc * phiM) / Y)
    return (-1 / (4 * np.pi)) * (np.log(1 + kl) - kl + .5 * kl * kl)
def s_1comp(x):
    ###THIS IS FROM LIN TO SPEED UP AND AVOID ERRORS
    return (x > epsilon)*x*np.log(x + (x < epsilon)) + 1e5*(x<0)
def entropy(phiM,protein):
    phiC = protein.qc * phiM + protein.phiS
    phiW = 1 - phiM - protein.phiS*2 - phiC
    #################FIGURE OUT LOGIC FOR 0s
    return s_1comp(phiM)/protein.N + s_1comp(protein.phiS)+ s_1comp(phiC) + s_1comp(phiW)

################FIRST DERIVATIVES##############################################################
def ds_1comp(x):
    ##ALSO FRM LIN FOR SPEED AND NO ERRORS
    return (np.log(x + 1e5 * (x <= 0)) + 1) * (x > 0)
def dfpintegrand(k,Y,phiM,protein):
    xe = xee(k,protein)
    g = gk(k,protein)
    c = ck(k,protein)
    rho = k * k * Y / (4 * np.pi) + protein.phiS*2 + protein.qc * phiM
    v2 = protein.w2*np.exp(-k*k/6)
    #testing fp with old version (fails crit finder)
    #v2 = 4*np.pi/3*np.exp(-k*k/6)
    c2 = c*c

    num = -2 * c2 * v2 * phiM + g * v2 * rho + 2 * xe * g * v2 * phiM + xe
    den = phiM * (c2*(-1*v2)*xe + xe*g*v2 +xe) + g * rho * v2 + rho

    return k*k*num/den
def d1_Frg_dphi(phiM,Y,protein):
    phic = protein.qc*phiM + protein.phiS
    phiW = 1 - phiM - phic - protein.phiS*2

    ###d1 entropy
    ds_dphi = (ds_1comp(phiM)/protein.N + protein.qc*ds_1comp(phic) - (1+protein.qc)*ds_1comp(phiW))*(phiM>0)

    ##d1 screening
    c = 4*np.pi/Y
    rho = protein.qc*phiM + protein.phiS
    k = np.sqrt(c*rho)

    temp = -k/2/(1+k)*(1/Y)
    dfion_dphi=temp*protein.qc*(phiM>0)

    #d1 fprotein
    upper,lower = 0,np.inf
    result = integrate.quad(dfpintegrand, lower, upper, args=(Y,phiM,protein), limit=iterlim)
    dfp_dphi = result[0] / (4 * np.pi * np.pi)

    d1_ftot = ds_dphi + dfion_dphi + dfp_dphi
    # w2 part from lili
    d1_ftot += protein.w2*phiM #d1f0

    #testing old version
    #d1_ftot += 4*np.pi*phiM/3

    if protein.W3_TOGGLE ==1:
        d1_ftot += 3*(protein.w3 - 1/6)*phiM**2
    if protein.crowding_toggle == 1:
        d1_ftot += -2 * protein.epsC * phiM

    return d1_ftot

#####################SECOND DERIVATIVE RG FREE ENERGIES############################################################
def d2s_1comp(x):
    return (x>0)/(x + (x==0))
def d2_FP_toint(k, Y, phiM,protein):
    phic = protein.qc*phiM + protein.phiS
    k2 = k*k
    rho = k2 * Y / (4 * np.pi) + protein.phiS + phic
    qc2, phi2, rho2 =  protein.qc * protein.qc, phiM * phiM, rho * rho
    xe = xee(k, protein)
    g = gk(k,protein)
    c = ck(k, protein)

    v2 = protein.w2 * np.exp(-1 * k2 / 6)
    #testing fp with old version (fails crit finder)
    #v2 = 4*np.pi/3*np.exp(-1*k2/6)
    D = xe * g - c * c
    vp2 = v2 * phi2*D / rho
    vp1= 2*v2*phiM*D/rho
    vr = xe/rho + v2*g

    num1 = 2*v2*D
    den = vp2 + phiM*vr + 1

    num2 = (vp1 + vr)*(vp1 + vr)
    return k2*(num1/(rho*den) - num2/(den*den))
def d2_Frg_phiM(phiM,Y,protein):
    qc = protein.qc
    phic = qc * phiM + protein.phiS
    phiW = 1 - phiM - phic - protein.phiS

    #################Entropyd2##########
    d2s = (d2s_1comp(phiM) / protein.N + qc * qc * d2s_1comp(phic) + (1 + qc) * (1 + qc) * d2s_1comp(phiW)) * (phiM > 0)

    #####d2Fion#################
    c = 4 * np.pi / Y
    rho = phic + protein.phiS
    k = np.sqrt((c) * (rho))
    ##THIS IS FROM LIN
    tp = qc / (1 + k) * (phiM > 0)
    temp = -1 * np.pi * (1 / Y) * (1 / Y) / (k + (k == 0)) * (k > 0)
    d2fion = temp * tp * tp

    # lili ML value
    d2f0 = protein.w2
    #testing old version
    #d2f0 = 4*np.pi/3

    #################Electrofreeenergyd2###########
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(d2_FP_toint, lowerlim, upperlim, args=(Y, phiM,protein), limit=iterlim,epsabs=epsabs, epsrel=epsrel)
    d2fp = result[0] / (4 * np.pi * np.pi)

    d2_ftot =  np.float64((d2s + d2fp + d2fion + d2f0))

    if protein.W3_TOGGLE == 1:
        d2_ftot += 6*(protein.w3 - 1/6)*phiM
    if protein.crowding_toggle == 1:
        d2_ftot += -2 * protein.epsC

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
        ures = root_scalar(d2f,x0=ui,x1=ui/5, rtol=epsabs, bracket = (1e-5, 1e4))
        if ures.converged:
            return np.float64(ures.root)
    except (ValueError,RuntimeError) as e:
        return np.nan

    # print('phi,t:' ,phiM, ures.root, 'attempted')
    # print(d2_Frg_phiM(phiM,ures.root,protein.phiS),'this is d2')
    #return np.float64(ures.root)
def findCrit(protein):
    bounds = (epsilon,1-epsilon)

    result = minimize_scalar(spin_yfromphi,args=(protein),method='bounded', bounds=bounds)
    (pf,uf) = (result.x, result.fun)
    tf = 1/uf
    return np.float64(pf), np.float64(tf)
def findSpins(Y,protein):
    ##THIS FROM LIN
    phiMax = 0.9
    #phiMax = 1-epsilon

    #print(Y,protein.Yc)
    phi1 = brenth(d2_Frg_phiM, epsilon, protein.phiC, args=(Y,protein))
    phi2 = brenth(d2_Frg_phiM, protein.phiC, .9,args = (Y,protein))
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

    #print(phi1spin, phi2spin, 'SPINS LEFT/RIGHT')

    phiB = (phi1spin+phi2spin)/2
    #phiB = protein.phiC

    assert np.isfinite(phi1spin), "phi1spin is not a finite number"
    assert np.isfinite(phi2spin), "phi2spin is not a finite number"

    ### GET CONSTRAINTS ###
    phiMax = (1-2*protein.phiS)/(1 + protein.qc) - epsilon ### FROM LIN ###
    bounds = [(epsilon, phi1spin - epsilon), (phi2spin + epsilon, phiMax-epsilon) ]
    #bounds = [(phi1spin/10, phi1spin - epsilon), (phi2spin*1.025*(protein.Yc/Y)**2 +epsilon, phi2spin*3*(protein.Yc/Y)**2)]
    t0 = time.time()

    #M = 'L-BFGS-B'
    #if protein.phiS ==0:
    M = 'SLSQP'
    #M= 'TNC'

    ### MAKE INITIAL ### IN PROGRESS ###
    #initial_guess= getInitialVsolved(Y,phi1spin,phi2spin,phiB,protein)
    initial_guess=(np.float64(phi1spin*.9),np.float64(phi2spin*1.1))#*(protein.Yc/Y) +epsilon))
    #initial_guess = (np.float64(phi1spin*0.9),np.float64(1.01*phi2spin*(protein.Yc/Y)**2.5))
    jac = Jac_rgRPA if protein.phiS == 0 else None
    maxL = minimize(FBINODAL,initial_guess,args=(Y,phiB,protein),method=M,jac=jac,bounds=bounds,options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20})#, 'maxfun':MINMAX})

    ### FINDING LOWEST SOLUTION ###
    phi1min = min(maxL.x)
    phi2min = max(maxL.x)

    ### PRINTING RESULTS FROM MINIMIZER ###
    v = (phi2min-phiB)/(phi2min-phi1min)

    print('MINIMIZER COMPLETE FOR Y = ',Y, 'MIN VALUES: phi1,phi2, v = ',phi1min,phi2min,v )
    #print('This step took ', time.time()-t0, 's\n')
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
