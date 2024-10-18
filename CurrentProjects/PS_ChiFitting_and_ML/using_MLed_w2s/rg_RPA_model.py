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
T0=100
iterlim=500
epsabs = 1e-15
epsrel = 1e-15
MINMAX=250
epsilon = 1e-18


###############STRUCTURE FACTORS#############################
def xee(k,x,protein):

    k2 = k * k
    exp_factor = -1 / 6 * x* k2
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.xeeSig[:protein.N] * exp_vals)
    return gksum / protein.N
def gk(k,x,protein):
    k2 = k * k
    exp_factor = -1 / 6 *x * k2
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.gkSig[:protein.N] * exp_vals)
    return gksum / protein.N
def ck(k,x,protein):
    k2 = k * k
    exp_factor = -1 / 6 * x* k2
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.ckSig[:protein.N] * exp_vals)
    return gksum / protein.N
def xee_r(k,x,protein):
    k2 = k * k
    exp_factor = -1 / 6 * x * k2
    i2 = protein.i_vals*protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.xeeSig[:protein.N] * exp_vals*i2)
    return gksum / protein.N
def gk_r(k,x,protein):
    k2 = k * k
    exp_factor = -1 / 6 * x * k2
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.gkSig[:protein.N] * exp_vals * i2)
    return gksum / protein.N
def ck_r(k,x,protein):
    # return np.mean(sigs* L2*np.exp((-1/6)*x*k*k*L))
    k2 = k * k
    exp_factor = -1 / 6 * x * k2
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.ckSig[:protein.N] * exp_vals * i2)
    return gksum / protein.N


def d1_xee(k,x,protein):
    k2 = k * k
    exp_factor = -1 / 6 *x* k2
    coeff = -k2 / 6
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.xeeSig[:protein.N ] * protein.i_vals * exp_vals * coeff)
    return gksum / protein.N
    # return k*k*k*k/36*np.mean(sigs*L2*L2*np.exp((-1/6)*x*k*k*L))
    #return -k * k / 6 * np.sum(sigS * L * np.exp((-1 / 6) * x * k * k * L))/N
def d1_gk(k,x,protein):
    k2 = k * k
    exp_factor = -1 / 6 *x * k2
    coeff = -k2 / 6
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.gkSig[:protein.N ] * protein.i_vals * exp_vals * coeff)
    return gksum / protein.N
def d1_ck(k,x,protein):
    k2 = k * k
    exp_factor = -1 / 6 *x * k2
    coeff = - k2 / 6
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.ckSig[:protein.N] * protein.i_vals * exp_vals * coeff)
    return gksum / protein.N
def d1_xee_r(k,x,protein):
    k2 = k * k
    exp_factor = -1 / 6 * x * k2
    coeff = -k2 / 6
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.xeeSig[:protein.N ] * protein.i_vals * exp_vals * i2 * coeff)
    return gksum / protein.N
def d1_gk_r(k,x,protein):
    k2 = k * k
    exp_factor = -1 / 6 * x * k2
    coeff = -k2 / 6
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.gkSig[:protein.N] * protein.i_vals * exp_vals * i2 * coeff)
    return gksum / protein.N
def d1_ck_r(k, x, protein):
    k2 = k * k
    exp_factor = -1 / 6 * x * k2
    coeff = -k2 / 6
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.ckSig[:protein.N ] * protein.i_vals * exp_vals * i2 * coeff)
    return gksum / protein.N


def d2_xee(k,x,protein):

    k2 = k * k
    exp_factor = -1 / 6 *x * k2
    coeff = k2 * k2 / 36
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.xeeSig[:protein.N]* exp_vals * i2 * coeff)
    return gksum / protein.N
def d2_gk(k,x,protein):
    # return k*k*k*k/36*np.mean(sigs*L2*L2*np.exp((-1/6)*x*k*k*L))
    k2 = k * k
    exp_factor = -1 / 6 *x * k2
    coeff = k2 * k2 / 36
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.gkSig[:protein.N] * exp_vals * i2 * coeff)
    return gksum / protein.N
def d2_ck(k,x,protein):
    # return k*k*k*k/36*np.mean(sigs*L2*L2*np.exp((-1/6)*x*k*k*L))
    k2 = k * k
    exp_factor = -1 / 6 *x * k2
    coeff = k2 * k2 / 36
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.ckSig[:protein.N ] * exp_vals * i2 * coeff)
    return gksum / protein.N
def d2_xee_r(k,x,protein):
    #return k*k*k*k/36*np.mean(sigS*L2*L2*np.exp((-1/6)*x*k*k*L))
    k2 = k*k
    exp_factor = -1/6*x*k2
    coeff = k2*k2/36
    i2 = protein.i_vals*protein.i_vals
    exp_vals = np.exp(exp_factor*protein.i_vals)
    gksum= np.sum(protein.xeeSig[:protein.N]*i2*exp_vals*i2*coeff)
    return gksum/protein.N
def d2_gk_r(k,x,protein):
    # return k*k*k*k/36*np.mean(sigS*L2*L2*np.exp((-1/6)*x*k*k*L))
    k2 = k * k
    exp_factor = -1 / 6 * x * k2
    coeff = k2 * k2 / 36
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.gkSig[:protein.N] * i2 * exp_vals * i2 * coeff)
    return gksum / protein.N
def d2_ck_r(k,x,protein):
    # return k*k*k*k/36*np.mean(sigs*L2*L2*np.exp((-1/6)*x*k*k*L))
    k2 = k * k
    exp_factor = -1 / 6 * x * k2
    coeff = k2 * k2 / 36
    i2 = protein.i_vals * protein.i_vals
    exp_vals = np.exp(exp_factor * protein.i_vals)
    gksum = np.sum(protein.ckSig[:protein.N] * i2 * exp_vals * i2 * coeff)
    return gksum / protein.N

### XSOLVERS ####
def x_solver(phiM,Y,protein):
    sln = brenth(x_eqn, 1/10000/protein.N, 1000*protein.N, args=(phiM,Y,protein))
    return sln
def x_eqn(x,phiM,Y,protein):
    inte = integrate.quad(x_eqn_toint,epsilon,np.inf,args=(phiM,Y,x,protein),limit=iterlim)[0]
    #print(inte , phiM, Y, x)
    eqn = 1 - 1/x - (protein.N/(18*(protein.N-1))) * inte
    return eqn
def x_eqn_toint(k,phiM,Y,x,protein):
    phic= phiM*protein.qc
    ex = xee(k,x,protein)
    ex_r = xee_r(k,x,protein)
    g = gk(k,x,protein)
    gkr = gk_r(k,x,protein)
    c = ck(k,x,protein)
    ckr = ck_r(k,x,protein)
    ionConst = k*k*Y/(4*np.pi) + protein.phiS + phic
    v2 = 4*np.pi/3

    num = ex_r + ionConst*gkr*v2 + phiM*v2*(ex_r*g + ex*gkr - 2*c*ckr)
    den = ionConst + phiM*(ex + g*ionConst*v2) + phiM*phiM*v2*(ex*g - c*c)
    return (num/den)*k*k*k*k/(2*np.pi*np.pi)

#d1
def d1_x_solver(phiM,Y,protein,x=None):
    x=x_solver(phiM,Y,protein) if x==None else x

    Nconst = protein.N/(18*(protein.N-1))
    lower,upper=0,np.inf
    I1,I2 = integrate.quad(d1_x_eqn_I1int,lower,upper, args=(phiM,Y,x,protein,),limit=iterlim)[0] ,integrate.quad(d1_x_eqn_I2int,lower,upper, args=(phiM,Y,x,protein,),limit=iterlim)[0]

    return Nconst*I1/((1/x/x)-Nconst*I2)
def d1_x_eqn_I1int(k,phiM,Y,x,protein):
    exr = xee_r(k, x, protein)
    ex = xee(k,x,protein)
    gkr = gk_r(k,x,protein)
    g = gk(k,x,protein)
    c = ck(k,x,protein)
    ckr = ck_r(k,x,protein)

    ionConst = k*k*Y/(4*np.pi) +protein.qc*phiM + protein.phiS
    r12 = ex*g - c*c
    m12 = ex*gkr + g*exr - 2*c*ckr
    v = 4*np.pi/3

    b0 = m12 + protein.qc*gkr
    c0 = phiM*(g*(ionConst)+ex/v) + ionConst/v + phiM*phiM*(r12)
    e0 = phiM*protein.qc*g + g*(ionConst) + 2*phiM*(r12) + (protein.qc+ex)/v
    f0 = (gkr*(ionConst)+phiM*(m12)+exr/v)

    return k*k*k*k*((b0*c0 - e0*f0)/(c0*c0))*(1/(2*np.pi*np.pi))
def d1_x_eqn_I2int(k,phiM,Y,x,protein):
    exr = xee_r(k, x, protein)
    ex = xee(k,x,protein)
    gkr = gk_r(k,x,protein)
    g = gk(k,x,protein)
    c = ck(k,x,protein)
    ckr = ck_r(k,x,protein)

    d1exr = d1_xee_r(k, x, protein)
    d1ex = d1_xee(k, x, protein)
    d1gkr = d1_gk_r(k, x,protein)
    d1g = d1_gk(k, x,protein)
    d1c = d1_ck(k, x, protein)
    d1ckr = d1_ck_r(k, x, protein)

    ionConst = k*k*Y/(4*np.pi) + protein.qc*phiM + protein.phiS
    v = 4*np.pi/3
    r12 = ex*g-c*c
    d12 = g*d1ex + ex*d1g-2*c*d1c
    phi2 = phiM*phiM

    a0 = d1gkr*(ionConst) + phiM*(gkr*d1ex + ex*d1gkr+ exr*d1g+g*d1exr-2*ckr*d1c-2*c*d1ckr) + d1exr/v
    c0 = phiM*(g*(ionConst)+ ex/v) + (ionConst/v) + phi2*(r12)
    d0 = phiM*d1g*(ionConst) + phiM*d1ex/v + phi2*(d12)
    f0 = (gkr*(ionConst)+ phiM*(ex*gkr+g*exr-2*c*ckr)+exr/v)

    return k*k*k*k*((a0*c0 - d0*f0)/(c0*c0))*(1/(2*np.pi*np.pi))

#d2
def d2_x_solver(phiM,Y,protein,x=None,dx=None):
    x=x_solver(phiM,Y,protein) if x==None else x
    dx=d1_x_solver(phiM,Y,protein,x) if dx==None else dx

    Nconst = protein.N/(18*(protein.N-1))
    lower,upper=0,np.inf
    I1,I2 = integrate.quad(d2_x_eqn_I1int,lower,upper, args=(phiM,Y,x,dx,protein,),limit=iterlim)[0] ,integrate.quad(d2_x_eqn_I2int,lower,upper, args=(phiM,Y,x,dx,protein,),limit=iterlim)[0]

    return (Nconst*I1 + 2*dx*dx/x/x/x)/(1/x/x - Nconst*I2)
def d2_x_eqn_I1int(k,phiM,Y,x,dx,protein):
    exr = xee_r(k, x, protein)
    ex = xee(k, x, protein)
    gkr = gk_r(k, x,protein)
    g = gk(k, x,protein)
    c = ck(k, x,protein)
    ckr = ck_r(k, x, protein)

    d1exr = d1_xee_r(k, x, protein)
    d1ex = d1_xee(k, x, protein)
    d1gkr = d1_gk_r(k, x,protein)
    d1g = d1_gk(k, x,protein)
    d1c = d1_ck(k, x, protein)
    d1ckr = d1_ck_r(k, x, protein)

    d2exr = d2_xee_r(k, x, protein)
    d2ex = d2_xee(k, x, protein)
    d2gkr = d2_gk_r(k, x,protein)
    d2g = d2_gk(k, x,protein)
    d2c = d2_ck(k, x, protein)
    d2ckr = d2_ck_r(k, x, protein)

    rho = k * k * Y / (4 * np.pi) + protein.phiS + protein.qc * phiM
    v = 4*np.pi/3
    DD = g*d1ex*dx + ex*d1g*dx - 2*c*d1c*dx
    r12 = ex*g-c*c
    phi2,dx2 = phiM*phiM, dx*dx
    ddel =protein.qc * g + d1ex * dx / v + d1g * dx * (rho)
    dm12 = gkr*d1ex*dx + exr*d1g*dx - 2*ckr*d1c*dx + g*d1exr*dx + ex*d1gkr*dx - 2*c*d1ckr*dx
    m12 = g*exr + ex*gkr - 2*c*ckr

    ##########DOUBLE CHECK THIS TERM!!!!!!!!!!!!!!
    a0 = (phi2 * (DD) + 2 * phiM * (r12) + phiM * (ddel) +  (protein.qc + ex)/v + g * (rho))#
    b0=  (m12 + protein.qc * gkr + d1exr * dx / v + d1gkr * dx * (rho) + phiM * (dm12))
    c0 = phi2 * (r12) + phiM * (ex / v + g * (rho)) +  (rho)/v
    d0 = (exr / v + gkr * (rho) + phiM * (m12))

    #f0 = phiM*phiM*g*d1ex + phiM*phiM*d1g*ex - 2*c*d1c*phiM*phiM - phiM*3*d1ex/(4*np.pi) - phiM*(rho+qc*phiM)*d1g
    g0 = (phi2 * (2*d1ex*d1g*dx2 - 2*d1c*d1c*dx2 + g*d2ex*dx2 + ex*d2g*dx2 - 2*c*d2c*dx2) + 4 * phiM * (DD) + phiM * (2 * protein.qc * d1g * dx + d2ex * dx2 / v + d2g * dx2 * (rho)) + 2 * protein.qc * g + 2 * (r12) + 2 * d1ex * dx / v + 2 * d1g * dx * (rho))
    #h0 = 3*ex/(4*np.pi)+d1gkr*(rho+qc*phiM) +phiM*(gkr*d1ex+exr*d1g-2*ckr*d1c+g*d1exr+ex*d1gkr-2*c*ckr)
    i0 = 2 * protein.qc * d1gkr * dx + 2 * (dm12) + d2exr * dx2 / v + d2gkr * dx2 * (rho) + phiM * (2 * d1g * d1exr * dx2 + 2 * d1ex * d1gkr * dx2 - 4 * d1c * d1ckr * dx2 + gkr * d2ex * dx2 + exr * d2g * dx2 - 2 * ckr * d2c * dx2 + g * d2exr * dx2 + ex * d2gkr * dx2 - 2 * c * d2ckr * dx2)

    return k*k*k*k*((2*d0*a0*a0 - d0*g0*c0 - 2*a0*b0*c0 + i0*c0*c0)/(c0*c0*c0))*(1/(2*np.pi*np.pi))
def d2_x_eqn_I2int(k, phiM, Y, x, dx,protein):
    exr = xee_r(k, x, protein)
    ex = xee(k, x, protein)
    gkr = gk_r(k, x,protein)
    g = gk(k, x,protein)
    c = ck(k, x, protein)
    ckr = ck_r(k, x, protein)

    d1exr = d1_xee_r(k, x, protein)
    d1ex = d1_xee(k, x, protein)
    d1gkr = d1_gk_r(k, x,protein)
    d1g = d1_gk(k, x,protein)
    d1c = d1_ck(k, x,protein)
    d1ckr = d1_ck_r(k,x,protein)

    k2,phi2 = k*k, phiM*phiM
    ionConst = k2 * Y / (4 * np.pi) + protein.phiS + protein.qc*phiM
    v = 4*np.pi/3
    r12 = ex*g - c*c
    m12 = g*exr + ex*gkr - 2*c*ckr

    c0 = phi2*(r12) + phiM*(ex/v + g*(ionConst)) + (ionConst)/v
    d0 = (exr/v + gkr*(ionConst) + phiM*(m12))
    f0 = phi2*(g*d1ex + d1g*ex - 2*c*d1c) + phiM*d1ex/v + phiM*(ionConst)*d1g
    h0 = d1exr/v + d1gkr*(ionConst) + phiM*(gkr*d1ex + exr*d1g - 2*ckr*d1c + g*d1exr + ex*d1gkr - 2*c*d1ckr)

    return k2 * k2 * ((h0 - d0*f0/c0)/c0) * (1 / (2 * np.pi * np.pi))


#################FREE ENERGIES#####################################
def ftot_rg(phiM, Y, protein,x=None):
    x=x_solver(phiM,Y,protein) if x==None else x

    if phiM > 1 or phiM < 0 or protein.phiS > 1 or protein.phiS < 0:
        print('illegal phi range detected')
        return np.nan

    ftot = entropy(phiM,protein) + fion(phiM, Y,protein) + rgFP(phiM, Y,protein,x)
    #this is where we add lili's predicted omega2 values.
    ftot += protein.w2*phiM*phiM/2

    #testing old version
    #ftot+= 2*np.pi*phiM*phiM/3

    if protein.W3_TOGGLE==1:
        ftot += (protein.w3 - 1/6)*phiM**3

    return ftot
def rgFPint(k,Y,phiM,protein,x):
    xe = xee(k, x, protein)
    g = gk(k, x,protein)
    c = ck(k, x, protein)
    v = protein.w2*np.exp(-1*k*k/6)
    nu = k*k*Y/4/np.pi + protein.qc*phiM + protein.phiS
    N1 = nu + phiM*(g*nu*v + xe) +v*phiM*phiM*(g*xe-c*c)
    A = N1/nu
    B= 1 + protein.Q/nu
    return (k*k/4/np.pi/np.pi)*np.log(A/B)
def rgFP(phiM, Y,protein,x=None):
    x = x_solver(phiM, Y,protein) if x==None else x

    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(rgFPint, lowerlim, upperlim, args=(Y, phiM,protein,x,), limit=iterlim, epsabs=epsabs,epsrel=epsrel)
    fp = result[0]
    return fp
def fion(phiM, Y,protein):
    kl = np.sqrt(4 * np.pi * (protein.phiS + protein.qc * phiM) / Y)
    return (-1 / (4 * np.pi)) * (np.log(1 + kl) - kl + .5 * kl * kl)
def s_1comp(x):
    ###THIS IS FROM LIN TO SPEED UP AND AVOID ERRORS
    return (x > epsilon)*x*np.log(x + (x < epsilon)) + 1e5*(x<0)
def entropy(phiM,protein):
    phiC = protein.qc * phiM
    phiW = 1 - phiM - protein.phiS - phiC
    #################FIGURE OUT LOGIC FOR 0s
    return s_1comp(phiM)/protein.N + s_1comp(protein.phiS)+ s_1comp(phiC) + s_1comp(phiW)

################FIRST DERIVATIVES##############################################################
def ds_1comp(x):
    ##ALSO FRM LIN FOR SPEED AND NO ERRORS
    return (np.log(x + 1e5 * (x <= 0)) + 1) * (x > 0)
def dfpintegrand(k,Y,phiM,protein,x,dx):
    xe = xee(k,x, protein)
    d1xe = d1_xee(k,x,protein)
    g = gk(k,x,protein)
    d1g = d1_gk(k,x,protein)
    c = ck(k,x,protein)
    d1c = d1_ck(k,x,protein)
    ionConst = k*k*Y/(4*np.pi) + protein.phiS + protein.qc*phiM
    v2 = protein.w2*np.exp(-k*k/6)
    r12 = xe*g-c*c
    vpp = v2*phiM*phiM

    num = (vpp*(g*d1xe*dx + xe*d1g*dx - 2*c*d1c*dx)/(ionConst)+ phiM*(d1xe*dx/ionConst- protein.qc*xe/ionConst/ionConst + v2*d1g*dx)- protein.qc*vpp*r12/(ionConst*ionConst)+2*v2*phiM*r12/ionConst + xe/ionConst +v2*g)
    den = vpp*r12/ionConst + phiM*(xe/ionConst + v2*g) + 1

    return num/den
def d1_Frg_dphi(phiM,Y,protein,x=None, dx = None):
    x = x_solver(phiM,Y,protein) if x==None else x
    dx = d1_x_solver(phiM,Y,protein,x) if dx ==None else dx
    phic = protein.qc*phiM
    phiW = 1 - phiM - phic - protein.phiS

    ###d1 entropy
    ds_dphi = (ds_1comp(phiM)/protein.N + protein.qc*ds_1comp(phic) - (1+protein.qc)*ds_1comp(phiW))*(phiM>0)

    ##d1 screening
    c = 4*np.pi/Y
    rho = protein.qc*phiM + protein.phiS
    k = np.sqrt(c*rho)

    temp = -k/2/(1+k)*(1/Y)
    dfion_dphi=temp*protein.qc*(phiM>0)
    #dfion_dphi= (-1*np.sqrt(np.pi)*qc*np.sqrt((phic + phiS)/Y))/(Y*(2*np.sqrt(np.pi)*np.sqrt((phic)/Y)+1))

    #d1 fprotein
    upper,lower = 0,np.inf
    result = integrate.quad(dfpintegrand, lower, upper, args=(Y,phiM,protein,x,dx), limit=iterlim)
    dfp_dphi = result[0] / (4 * np.pi * np.pi)

    d1_ftot = ds_dphi + dfion_dphi + dfp_dphi
    d1_ftot += protein.w2 * phiM

    if protein.W3_TOGGLE == 1:
        d1_ftot += 3 * (protein.w3 - 1 / 6) * phiM ** 2

    return d1_ftot

#####################SECOND DERIVATIVE RG FREE ENERGIES############################################################
def d2s_1comp(x):
    return (x>0)/(x + (x==0))
def d2_FP_toint(k, Y, phiM,protein,x,dx,ddx):
    phic = protein.qc*phiM
    k2 = k*k
    rho = k2 * Y / (4 * np.pi) + protein.phiS + phic
    dx2, qc2, phi2, rho2 = dx * dx, protein.qc * protein.qc, phiM * phiM, rho * rho
    xe = xee(k, x, protein)
    d1xe = d1_xee(k, x, protein)
    d1xe_x = d1xe*dx
    d2xe = d2_xee(k, x, protein)*dx2 + d1xe*ddx
    g = gk(k, x,protein)
    d1g = d1_gk(k, x,protein)
    d1g_x = d1g*dx
    d2g = d2_gk(k, x,protein)*dx2 + d1g*ddx
    c = ck(k, x, protein)
    d1c = d1_ck(k, x, protein)
    d1c_x = d1c*dx
    d2c = d2_ck(k, x, protein)*dx2 + d1c*ddx

    v2 = protein.w2 * np.exp(-1 * k2 / 6)
    D2BIG = g * d2xe + 2 * d1xe_x * d1g_x + xe * d2g - 2 * c * d2c - 2 * d1c_x * d1c_x
    D = xe * g - c * c
    DD = g * d1xe_x + xe * d1g_x - 2 * c * d1c_x
    vp22 = v2 * phi2 / rho2
    vp21 = v2*phi2/rho
    vr = xe/rho + v2*g

    Num1 = (-2 * protein.qc * vp22* (DD) + 4 * vp21 * (DD) / phiM + 2 * d1xe_x / rho + phiM * (d2xe/ rho - 2 * protein.qc * d1xe_x / (rho2) + 2 * qc2 * xe / (rho * rho2) + v2 * (d2g)) + vp21 * (D2BIG) + 2 * qc2 * vp22 * (D) / (rho) - 4 * protein.qc * vp22 * (D) / (phiM) + 2 * v2 * D / rho - 2 * protein.qc * xe / (rho2) + 2 * v2 * d1g_x)
    Den = (vp21*D + phiM * (vr) + 1)
    Num2 = (vp21*DD + phiM * (d1xe_x / rho - protein.qc * xe / (rho2) + v2 * d1g_x) - protein.qc * vp22 * D + 2 * vp21 * D / phiM + vr)

    return k2 * (Num1 / Den - (Num2 * Num2) /Den/Den )
def d2_Frg_phiM(phiM,Y,protein,x=None,dx=None,ddx=None):
    x = x_solver(phiM, Y,protein) if x == None else x
    dx = d1_x_solver(phiM, Y, protein,x) if dx == None else dx
    ddx = d2_x_solver(phiM, Y, protein,x, dx) if ddx == None else ddx
    phic = protein.qc * phiM
    phiW = 1 - phiM - phic - protein.phiS

    #################Entropyd2##########
    d2s = (d2s_1comp(phiM) / protein.N + protein.qc * protein.qc * d2s_1comp(phic) + (1 + protein.qc) * (1 + protein.qc) * d2s_1comp(phiW)) * (phiM > 0)

    #####d2Fion#################
    c = 4 * np.pi / Y
    rho = phic + protein.phiS
    k = np.sqrt((c) * (rho))
    ##THIS IS FROM LIN
    tp = protein.qc / (1 + k) * (phiM > 0)
    temp = -1 * np.pi * (1 / Y) * (1 / Y) / (k + (k == 0)) * (k > 0)
    d2fion = temp * tp * tp

    # d2f0
    d2f0 = protein.w2

    #################Electrofreeenergyd2###########
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(d2_FP_toint, lowerlim, upperlim, args=(Y, phiM,protein, x, dx, ddx), limit=iterlim)
    d2fp = result[0] / (4 * np.pi * np.pi)

    d2_ftot =  np.float64((d2s + d2fp + d2fion + d2f0))

    #### FH OPTION ###
    if protein.W3_TOGGLE == 1:
        d2_ftot += 6 * (protein.w3 - 1 / 6) * phiM
    #print(d2s,d2fp,d2fion,d2f0,'this is d2s, d2fp, d2fion, d2f0 at phi= ',phiM,Y)
    print(d2_ftot,'this is d2 at phi,Y= ',phiM,Y)

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
        ures = root_scalar(d2f,x0=ui,x1=ui/2, rtol=epsabs, bracket = (1e-4, 1e3))
        #ures = fsolve(d2f,ui,xtol=epsabs)
        print('phi,t:' ,phiM, 1/ures.root, 'attempted')

        if ures.converged:
            return np.float64(ures.root)
    except (ValueError,RuntimeError) as e:
        return np.nan

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

    #phiB = (phi1spin+phi2spin)/2
    phiB = protein.phiC

    assert np.isfinite(phi1spin), "phi1spin is not a finite number"
    assert np.isfinite(phi2spin), "phi2spin is not a finite number"

    ### GET CONSTRAINTS ###
    phiMax = (1-2*protein.phiS)/(1 + protein.qc) - epsilon ### FROM LIN ###
    bounds = [(epsilon, phi1spin - epsilon), (phi2spin + epsilon, phiMax-epsilon) ]
    #bounds = [(phi1spin/10, phi1spin - epsilon), (phi2spin*1.025*(protein.Yc/Y)**2 +epsilon, phi2spin*3*(protein.Yc/Y)**2)]
    t0 = time.time()

    #M = 'L-BFGS-B'
    M = 'SLSQP'

    ### MAKE INITIAL ### IN PROGRESS ###
    #initial_guess= getInitialVsolved(Y,phi1spin,phi2spin,phiB,protein)
    initial_guess=(np.float64(phi1spin*.9-epsilon),np.float64(phi2spin*1.1*(protein.Yc/Y) +epsilon))
    #initial_guess = (np.float64(phi1spin*0.9),np.float64(1.01*phi2spin*(protein.Yc/Y)**2.5))

    maxL = minimize(FBINODAL, initial_guess, args=(Y, phiB,protein), method=M, jac=Jac_rgRPA, bounds=bounds , options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20})#, 'maxfun':MINMAX})
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
