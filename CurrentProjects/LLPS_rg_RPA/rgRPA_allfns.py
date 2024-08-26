import numpy as np
from OldProteinProjects.SCDcalc import *
from scipy import integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize,minimize_scalar
from scipy.optimize import root_scalar
from rgRPA_init import *
from scipy.optimize import brenth, brent
import math
import time

########################ConstANTS################################
T0=1e8
iterlim=2500
MINMAX=25
qL = np.array(qs)
Q = np.sum(qL*qL)/N
L = np.arange(N)
L2 = L*L

def getSigShifts(qs):
    sigSij = []
    sigGs = []
    for i in range(len(qs)):  ###abs(tau - mu)
        sigij = 0
        sigG=0
        for j in range(len(qs)-1):  #### tau (starting spot)
            if (j+i)<= len(qs)-1:
                sigij += qs[j] * qs[j + i] #
                sigG += 1
        if i ==0:
            sigSij.append(sigij)
            sigGs.append(sigG+1)
        if i!=0:
            sigSij.append(2 * sigij)
            sigGs.append(2 *sigG )

    #######THIS IS FROM LIN GITHUB, COULDN'T FIGURE OUT MY SUM METHOD
    mlx = np.kron(qs, np.ones(N)).reshape((N, N))
    sigSi = np.array([np.sum(mlx.diagonal(n) + mlx.diagonal(-n)) for n in range(N)])
    sigSi[0] /= 2

    return sigSij ,sigSi ,sigGs
sigShift_xe, sigShift_ck ,sigShift_gk = getSigShifts(qs)

##############STRUCTURE FACTORS & DERIVATIVES###########################

###############STRUCTURE FACTORS#############################
def xee(k,x,sigS):

    ### THIS IS MY OLD VERSION. SAVING COMMENTED OUT TO SHOW WHAT LINS IS DOING
    ### THE MEAN VERSION IS MUCH MUCH FASTER AND HAS REPLACED MY CODE
    # xeesum=0
    # for i in range(0,N-1):
    #     xeesum += sigS[i]*np.exp((-1/6)*x*(k*k)*i)
    # return xeesum/N

    return np.mean(sigS*np.exp((-1/6)*x*k*k*L))
def xee_r(k,x,sigS):
    return np.mean(sigS*L2*np.exp((-1/6)*x*k*k*L))
def gk(k,x):
    return np.mean(sigShift_gk*np.exp((-1/6)*x*k*k*L))
def gk_r(k,x):
    return np.mean(sigShift_gk*L2*np.exp((-1/6)*x*k*k*L))
def ck(k,x,sigs):
    return np.mean(sigs*np.exp((-1/6)*x*k*k*L))
def ck_r(k,x,sigs):
    return np.mean(sigs*L2*np.exp((-1/6)*k*k*x*L))

#d1
def d1_xee(k,x,sigS):
    # xeesum=0
    # for i in range(0,N-1):
    #     xeesum += sigS[i]*np.exp((-1/6)*x*(k*k)*i) *(-1/6)*(k*k)*i
    # return xeesum/N
    return -k*k/6*np.mean(sigS*L*np.exp((-1/6)*x*k*k*L))
def d1_xee_r(k,x,sigS):
    return -k*k/6*np.mean(sigS*L*L2*np.exp((-1/6)*x*k*k*L))
def d1_gk(k,x):
    return -k*k/6*np.mean(sigShift_gk*L*np.exp((-1/6)*x*k*k*L))
def d1_gk_r(k,x):
    return -k*k/6*np.mean(sigShift_gk*L2*L*np.exp((-1/6)*x*k*k*L))
def d1_ck(k,x,sigs):
    return -k*k/6*np.mean(sigs*L*np.exp((-1/6)*x*k*k*L))
def d1_ck_r(k,x,sigs):
    return -k*k/6*np.mean(sigs*L2*L*np.exp((-1/6)*x*k*k*L))

#d2
def d2_xee(k,x,sigS):
    # xeesum=0
    # for i in range(0,N-1):
    #     xeesum += sigS[i]*np.exp((-1/6)*x*(k*k)*i) *(1/36)*(k*k)*i*(k*k)*i
    # return xeesum/N
    return k*k*k*k/36*np.mean(sigS*L2*np.exp((-1/6)*x*k*k*L))
def d2_xee_r(k,x,sigS):
    return k*k*k*k/36*np.mean(sigS*L2*L2*np.exp((-1/6)*x*k*k*L))
def d2_gk(k,x):
    return k*k*k*k/36*np.mean(sigShift_gk*L2*np.exp((-1/6)*x*k*k*L))
def d2_gk_r(k,x):
    return k*k*k*k/36*np.mean(sigShift_gk*L2*L2*np.exp((-1/6)*x*k*k*L))
def d2_ck(k,x,sigs):
    return k*k*k*k/36*np.mean(sigs*L2*np.exp((-1/6)*x*k*k*L))
def d2_ck_r(k,x,sigs):
    return k*k*k*k/36*np.mean(sigs*L2*L2*np.exp((-1/6)*x*k*k*L))

############SOLVING FOR X #####################################################
def x_solver(phiM,Y):
    #print(x_eqn(1/(N*1000),phiM,Y),'lowbound', x_eqn(100*N,phiM,Y), 'upper')

    sln = brenth(x_eqn, 1/100/N, 100*N, args=(phiM,Y))
    #sln = fsolve(x_eqn,np.array([.5]),args=(phiM,Y,))
    #sln = root_scalar(x_eqn,x0=.5, args=(phiM,Y,))
    #bounds=[(0,1)]
    #sln = minimize(x_eqn, np.array([.5]),args=(phiM,Y,),method='Nelder-Mead',bounds=bounds)
    return sln
def x_eqn(x,phiM,Y):
    inte = integrate.quad(x_eqn_toint,epsilon,np.inf,args=(phiM,Y,x),limit=iterlim)[0]
    #print(inte , phiM, Y, x)
    eqn = 1 - 1/x - (N/(18*(N-1))) * inte
    if math.isnan(eqn):
        if x<.01:
            eqn= -1e10
        elif x> .2:
            eqn = .99999999999
    return eqn
    #return np.sqrt(eqn*eqn)
def x_eqn_toint(k,phiM,Y,x):
    phic= phiM*qc
    ex = xee(k,x,sigS=sigShift_xe)
    ex_r = xee_r(k,x,sigS=sigShift_xe)
    g = gk(k,x)
    gkr = gk_r(k,x)
    c = ck(k,x,sigs=sigShift_ck)
    ckr = ck_r(k,x,sigs= sigShift_ck)
    ionConst = k*k*Y/(4*np.pi) + phiS + phic
    v2 = 4*np.pi/3

    num = ex_r + ionConst*gkr*v2 + phiM*v2*(ex_r*g + ex*gkr - 2*c*ckr)
    den = ionConst + phiM*(ex + g*ionConst*v2) + phiM*phiM*v2*(ex*g - c*c)
    return (num/den)*k*k*k*k/(2*np.pi*np.pi)

#d1
def d1_x_solver(phiM,Y,x=None):
    x=x_solver(phiM,Y) if x==None else x

    Nconst = N/(18*(N-1))
    lower,upper=0,np.inf
    I1,I2 = integrate.quad(d1_x_eqn_I1int,lower,upper, args=(phiM,Y,x,),limit=iterlim)[0] ,integrate.quad(d1_x_eqn_I2int,lower,upper, args=(phiM,Y,x,),limit=iterlim)[0]

    return Nconst*I1/((1/x/x)-Nconst*I2)
def d1_x_eqn_I1int(k,phiM,Y,x):
    exr = xee_r(k, x, sigS=sigShift_xe)
    ex = xee(k,x,sigS=sigShift_xe)
    gkr = gk_r(k,x)
    g = gk(k,x)
    c = ck(k,x,sigs=sigShift_ck)
    ckr = ck_r(k,x,sigs= sigShift_ck)

    ionConst = k*k*Y/(4*np.pi) +qc*phiM + phiS
    r12 = ex*g - c*c
    m12 = ex*gkr + g*exr - 2*c*ckr
    v = 4*np.pi/3

    b0 = m12 + qc*gkr
    c0 = phiM*(g*(ionConst)+ex/v) + ionConst/v + phiM*phiM*(r12)
    e0 = phiM*qc*g + g*(ionConst) + 2*phiM*(r12) + (qc+ex)/v
    f0 = (gkr*(ionConst)+phiM*(m12)+exr/v)

    return k*k*k*k*((b0*c0 - e0*f0)/(c0*c0))*(1/(2*np.pi*np.pi))
def d1_x_eqn_I2int(k,phiM,Y,x):
    exr = xee_r(k, x, sigS=sigShift_xe)
    ex = xee(k,x,sigS=sigShift_xe)
    gkr = gk_r(k,x)
    g = gk(k,x)
    c = ck(k,x,sigs=sigShift_ck)
    ckr = ck_r(k,x,sigs= sigShift_ck)

    d1exr = d1_xee_r(k, x, sigS=sigShift_xe)
    d1ex = d1_xee(k, x, sigS=sigShift_xe)
    d1gkr = d1_gk_r(k, x)
    d1g = d1_gk(k, x)
    d1c = d1_ck(k, x, sigs=sigShift_ck)
    d1ckr = d1_ck_r(k, x, sigs=sigShift_ck)

    ionConst = k*k*Y/(4*np.pi) + qc*phiM + phiS
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
def d2_x_solver(phiM,Y,x=None,dx=None):
    x=x_solver(phiM,Y) if x==None else x
    dx=d1_x_solver(phiM,Y ,x) if dx==None else dx

    Nconst = N/(18*(N-1))
    lower,upper=0,np.inf
    I1,I2 = integrate.quad(d2_x_eqn_I1int,lower,upper, args=(phiM,Y,x,dx,),limit=iterlim)[0] ,integrate.quad(d2_x_eqn_I2int,lower,upper, args=(phiM,Y,x,dx,),limit=iterlim)[0]

    return (Nconst*I1 + 2*dx*dx/x/x/x)/(1/x/x - Nconst*I2)
def d2_x_eqn_I1int(k,phiM,Y,x,dx):
    exr = xee_r(k, x, sigS=sigShift_xe)
    ex = xee(k, x, sigS=sigShift_xe)
    gkr = gk_r(k, x)
    g = gk(k, x)
    c = ck(k, x, sigs=sigShift_ck)
    ckr = ck_r(k, x, sigs=sigShift_ck)

    d1exr = d1_xee_r(k, x, sigS=sigShift_xe)
    d1ex = d1_xee(k, x, sigS=sigShift_xe)
    d1gkr = d1_gk_r(k, x)
    d1g = d1_gk(k, x)
    d1c = d1_ck(k, x, sigs=sigShift_ck)
    d1ckr = d1_ck_r(k, x, sigs=sigShift_ck)

    d2exr = d2_xee_r(k, x, sigS=sigShift_xe)
    d2ex = d2_xee(k, x, sigS=sigShift_xe)
    d2gkr = d2_gk_r(k, x)
    d2g = d2_gk(k, x)
    d2c = d2_ck(k, x, sigs=sigShift_ck)
    d2ckr = d2_ck_r(k, x, sigs=sigShift_ck)

    rho = k * k * Y / (4 * np.pi) + phiS + qc * phiM
    v = 4*np.pi/3
    DD = g*d1ex*dx + ex*d1g*dx - 2*c*d1c*dx
    r12 = ex*g-c*c
    phi2,dx2 = phiM*phiM, dx*dx
    ddel =qc * g + d1ex * dx / v + d1g * dx * (rho)
    dm12 = gkr*d1ex*dx + exr*d1g*dx - 2*ckr*d1c*dx + g*d1exr*dx + ex*d1gkr*dx - 2*c*d1ckr*dx
    m12 = g*exr + ex*gkr - 2*c*ckr

    ##########DOUBLE CHECK THIS TERM!!!!!!!!!!!!!!
    a0 = (phi2 * (DD) + 2 * phiM * (r12) + phiM * (ddel) +  (qc + ex)/v + g * (rho))#
    b0=  (m12 + qc * gkr + d1exr * dx / v + d1gkr * dx * (rho) + phiM * (dm12))
    c0 = phi2 * (r12) + phiM * (ex / v + g * (rho)) +  (rho)/v
    d0 = (exr / v + gkr * (rho) + phiM * (m12))

    #f0 = phiM*phiM*g*d1ex + phiM*phiM*d1g*ex - 2*c*d1c*phiM*phiM - phiM*3*d1ex/(4*np.pi) - phiM*(rho+qc*phiM)*d1g
    g0 = (phi2 * (2*d1ex*d1g*dx2 - 2*d1c*d1c*dx2 + g*d2ex*dx2 + ex*d2g*dx2 - 2*c*d2c*dx2) + 4 * phiM * (DD) + phiM * (2 * qc * d1g * dx + d2ex * dx2 / v + d2g * dx2 * (rho)) + 2 * qc * g + 2 * (r12) + 2 * d1ex * dx / v + 2 * d1g * dx * (rho))
    #h0 = 3*ex/(4*np.pi)+d1gkr*(rho+qc*phiM) +phiM*(gkr*d1ex+exr*d1g-2*ckr*d1c+g*d1exr+ex*d1gkr-2*c*ckr)
    i0 = 2 * qc * d1gkr * dx + 2 * (dm12) + d2exr * dx2 / v + d2gkr * dx2 * (rho) + phiM * (2 * d1g * d1exr * dx2 + 2 * d1ex * d1gkr * dx2 - 4 * d1c * d1ckr * dx2 + gkr * d2ex * dx2 + exr * d2g * dx2 - 2 * ckr * d2c * dx2 + g * d2exr * dx2 + ex * d2gkr * dx2 - 2 * c * d2ckr * dx2)

    return k*k*k*k*((2*d0*a0*a0 - d0*g0*c0 - 2*a0*b0*c0 + i0*c0*c0)/(c0*c0*c0))*(1/(2*np.pi*np.pi))
def d2_x_eqn_I2int(k, phiM, Y, x, dx):
    exr = xee_r(k, x, sigS=sigShift_xe)
    ex = xee(k, x, sigS=sigShift_xe)
    gkr = gk_r(k, x)
    g = gk(k, x)
    c = ck(k, x, sigs=sigShift_ck)
    ckr = ck_r(k, x, sigs=sigShift_ck)

    d1exr = d1_xee_r(k, x, sigS=sigShift_xe)
    d1ex = d1_xee(k, x, sigS=sigShift_xe)
    d1gkr = d1_gk_r(k, x)
    d1g = d1_gk(k, x)
    d1c = d1_ck(k, x, sigs=sigShift_ck)
    d1ckr = d1_ck_r(k,x,sigs = sigShift_ck)

    k2,phi2 = k*k, phiM*phiM
    ionConst = k2 * Y / (4 * np.pi) + phiS + qc*phiM
    v = 4*np.pi/3
    r12 = ex*g - c*c
    m12 = g*exr + ex*gkr - 2*c*ckr

    c0 = phi2*(r12) + phiM*(ex/v + g*(ionConst)) + (ionConst)/v
    d0 = (exr/v + gkr*(ionConst) + phiM*(m12))
    f0 = phi2*(g*d1ex + d1g*ex - 2*c*d1c) + phiM*d1ex/v + phiM*(ionConst)*d1g
    h0 = d1exr/v + d1gkr*(ionConst) + phiM*(gkr*d1ex + exr*d1g - 2*ckr*d1c + g*d1exr + ex*d1gkr - 2*c*d1ckr)

    return k2 * k2 * ((h0 - d0*f0/c0)/c0) * (1 / (2 * np.pi * np.pi))

################TEST#######################
# x = x_solver(.3,.1)
# dx = d1_x_solver(.3,.1,x)
# ddx = d2_x_solver(.3,.1,x,dx)
# print(x,dx,ddx)

#################FREE ENERGIES#####################################
def ftot_rg(phiM, Y, phiS, x=None):
    x=x_solver(phiM,Y) if x==None else x

    if phiM > 1 or phiM < 0 or phiS > 1 or phiS < 0:
        print('illegal phi range detected')
        return np.nan
    return entropy(phiM, phiS) + fion(phiM, Y, phiS) + rgFP(phiM, Y, phiS, x) + 2*np.pi*phiM*phiM/3  ##f0 term
def rgFPint(k,Y,phiM,phiS,x):
    xe = xee(k, x, sigS=sigShift_xe)
    g = gk(k, x)
    c = ck(k, x, sigs=sigShift_ck)
    v = (4*np.pi/3)*np.exp(-1*k*k/6)
    nu = k*k*Y/4/np.pi + qc*phiM + phiS
    N1 = nu + phiM*(g*nu*v + xe) +v*phiM*phiM*(g*xe-c*c)
    A = N1/nu
    B= 1 + Q/nu
    return (k*k/4/np.pi/np.pi)*np.log(A/B)
def rgFP(phiM, Y, phiS,x=None):
    x = x_solver(phiM, Y) if x==None else x

    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(rgFPint, lowerlim, upperlim, args=(Y, phiM, phiS,x), limit=iterlim)
    fp = result[0]
    return fp
def fion(phiM, Y, phiS):
    kl = np.sqrt(4 * np.pi * (phiS + qc * phiM) / Y)
    return (-1 / (4 * np.pi)) * (np.log(1 + kl) - kl + .5 * kl * kl)
def s_1comp(x):
    ###THIS IS FROM LIN TO SPEED UP AND AVOID ERRORS
    return (x > epsilon)*x*np.log(x + (x < epsilon)) + 1e5*(x<0)
def entropy(phiM, phiS):
    phiC = qc * phiM
    phiW = 1 - phiM - phiS - phiC
    #################FIGURE OUT LOGIC FOR 0s
    return s_1comp(phiM)/N + s_1comp(phiS)+ s_1comp(phiC) + s_1comp(phiW)

################FIRST DERIVATIVES##############################################################
def ds_1comp(x):
    ##ALSO FRM LIN FOR SPEED AND NO ERRORS
    return (np.log(x + 1e5 * (x <= 0)) + 1) * (x > 0)
def dfpintegrand(k,Y,phiM,phiS,x,dx):
    xe = xee(k,x, sigS=sigShift_xe)
    d1xe = d1_xee(k,x,sigS=sigShift_xe)
    g = gk(k,x)
    d1g = d1_gk(k,x)
    c = ck(k,x,sigs=sigShift_ck)
    d1c = d1_ck(k,x,sigs=sigShift_ck)
    ionConst = k*k*Y/(4*np.pi) + phiS + qc*phiM
    v2 = (4*np.pi/3)*np.exp(-k*k/6)
    r12 = xe*g-c*c
    vpp = v2*phiM*phiM

    num = (vpp*(g*d1xe*dx + xe*d1g*dx - 2*c*d1c*dx)/(ionConst)+ phiM*(d1xe*dx/ionConst- qc*xe/ionConst/ionConst + v2*d1g*dx)- qc*vpp*r12/(ionConst*ionConst)+2*v2*phiM*r12/ionConst + xe/ionConst +v2*g)
    den = vpp*r12/ionConst + phiM*(xe/ionConst + v2*g) + 1

    return num/den
def d1_Frg_dphi(phiM,Y,phiS,x=None, dx = None):
    x = x_solver(phiM,Y) if x==None else x
    dx = d1_x_solver(phiM,Y,x) if dx ==None else dx
    phic = qc*phiM
    phiW = 1 - phiM - phic - phiS

    ###d1 entropy
    ds_dphi = (ds_1comp(phiM)/N + qc*ds_1comp(phic) - (1+qc)*ds_1comp(phiW))*(phiM>0)

    ##d1 screening
    c = 4*np.pi/Y
    rho = qc*phiM + phiS
    k = np.sqrt(c*rho)

    temp = -k/2/(1+k)*(1/Y)
    dfion_dphi=temp*qc*(phiM>0)
    #dfion_dphi= (-1*np.sqrt(np.pi)*qc*np.sqrt((phic + phiS)/Y))/(Y*(2*np.sqrt(np.pi)*np.sqrt((phic)/Y)+1))

    #d1 fprotein
    upper,lower = 0,np.inf
    result = integrate.quad(dfpintegrand, lower, upper, args=(Y,phiM,phiS,x,dx), limit=iterlim)
    dfp_dphi = result[0] / (4 * np.pi * np.pi)

    return ds_dphi+dfion_dphi+dfp_dphi + 4*np.pi*phiM/3 #d1f0

#####################SECOND DERIVATIVE RG FREE ENERGIES############################################################
def d2s_1comp(x):
    return (x>0)/(x + (x==0))
def d2_FP_toint(k, Y, phiM,x,dx,ddx):
    phic = qc*phiM
    k2 = k*k
    rho = k2 * Y / (4 * np.pi) + phiS + phic
    dx2, qc2, phi2, rho2 = dx * dx, qc * qc, phiM * phiM, rho * rho
    xe = xee(k, x, sigS=sigShift_xe)
    d1xe = d1_xee(k, x, sigS=sigShift_xe)
    d1xe_x = d1xe*dx
    d2xe = d2_xee(k, x, sigS=sigShift_xe)*dx2 + d1xe*ddx
    g = gk(k, x)
    d1g = d1_gk(k, x)
    d1g_x = d1g*dx
    d2g = d2_gk(k, x)*dx2 + d1g*ddx
    c = ck(k, x, sigs=sigShift_ck)
    d1c = d1_ck(k, x, sigs=sigShift_ck)
    d1c_x = d1c*dx
    d2c = d2_ck(k, x, sigs=sigShift_ck)*dx2 + d1c*ddx

    v2 = (4 * np.pi / 3) * np.exp(-1 * k2 / 6)
    D2BIG = g * d2xe + 2 * d1xe_x * d1g_x + xe * d2g - 2 * c * d2c - 2 * d1c_x * d1c_x
    D = xe * g - c * c
    DD = g * d1xe_x + xe * d1g_x- 2 * c * d1c_x
    vp22 = v2 * phi2 / rho2
    vp21 = v2*phi2/rho
    vr = xe/rho + v2*g

    Num1 = (-2 * qc * vp22* (DD) + 4 * vp21 * (DD) / phiM + 2 * d1xe_x / rho + phiM * (d2xe/ rho - 2 * qc * d1xe_x / (rho2) + 2 * qc2 * xe / (rho * rho2) + v2 * (d2g)) + vp21 * (D2BIG) + 2 * qc2 * vp22 * (D) / (rho) - 4 * qc * vp22 * (D) / (phiM) + 2 * v2 * D / rho - 2 * qc * xe / (rho2) + 2 * v2 * d1g_x)
    Den = (vp21*D + phiM * (vr) + 1)
    Num2 = (vp21*DD + phiM * (d1xe_x / rho - qc * xe / (rho2) + v2 * d1g_x) - qc * vp22 * D + 2 * vp21 * D / phiM + vr)

    return k2 * (Num1 / Den - (Num2 * Num2) /Den/Den )
def d2_Frg_Y(Y,phiM,phiS,x = None, dx = None, ddx = None):
    x = x_solver(phiM, Y) if x == None else x
    dx = d1_x_solver(phiM, Y,x) if dx == None else dx
    ddx = d2_x_solver(phiM, Y,x,dx) if ddx == None else ddx
    phic = qc*phiM
    phiW = 1 - phiM - phic - phiS
    #################Entropyd2##########
    d2s = (d2s_1comp(phiM)/N + qc*qc*d2s_1comp(phic) + (1+qc)*(1+qc)*d2s_1comp(phiW))*(phiM>0)

    #####d2Fion#################
    rho = phic + phiS
    k = np.sqrt((4*np.pi/Y)*(rho))
    ##THIS IS FROM LIN
    tp = qc/(1+k)*(phiM>0)
    temp = -np.pi*(1/Y)*(1/Y)/(k + (k==0))*(k>0)
    d2fion = temp*tp*tp

    #################Electrofreeenergyd2###########
    result = integrate.quad(d2_FP_toint, 0, np.inf, args=(Y, phiM,x,dx,ddx), limit=iterlim)

    d2fp = result[0] / (4 * np.pi * np.pi)
    #print(d2s+d2f0,d2fp,d2fion, Y)
    return np.float64((d2s + d2fp + d2fion + 4*np.pi/3)) #d2f0
def d2_Frg_phiM(phiM,Y,phiS,x=None,dx=None,ddx=None):
    x = x_solver(phiM, Y) if x == None else x
    dx = d1_x_solver(phiM, Y, x) if dx == None else dx
    ddx = d2_x_solver(phiM, Y, x, dx) if ddx == None else ddx
    phic = qc * phiM
    phiW = 1 - phiM - phic - phiS
    #################Entropyd2##########
    d2s = (d2s_1comp(phiM) / N + qc * qc * d2s_1comp(phic) + (1 + qc) * (1 + qc) * d2s_1comp(phiW)) * (phiM > 0)

    #####d2Fion#################
    c = 4 * np.pi / Y
    rho = phic + phiS
    k = np.sqrt((c) * (rho))
    ##THIS IS FROM LIN
    tp = qc / (1 + k) * (phiM > 0)
    temp = -np.pi * (1 / Y) * (1 / Y) / (k + (k == 0)) * (k > 0)
    d2fion = temp * tp * tp

    # d2f0
    d2f0 = 4 * np.pi / 3

    #################Electrofreeenergyd2###########
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(d2_FP_toint, lowerlim, upperlim, args=(Y, phiM, x, dx, ddx), limit=iterlim)

    d2fp = result[0] / (4 * np.pi * np.pi)
    # print(d2s+d2f0,d2fp,d2fion, Y)
    return np.float64((d2s + d2fp + d2fion + d2f0))

###############################################################################################################
#######SOLVER METHODS BELOW####################################################################################
###############################################################################################################

def getSpinodalrg(phiMs):
    Ys=[]
    for j in range(len(phiMs)):
        i = phiMs[j]
        y = brenth(d2_Frg_Y, .01, 10, args=(i, phiS,))
        print('phi', i, 'l/lb', y)
        Ys.append(y)
    return Ys
def spin_yfromphi(phiM,phiS,guess):
    guess1 = guess

    ans = root_scalar(d2_Frg_Y, bracket=[1/N/1000, 10*N], args=(phiM,phiS),method='brenth',x0=guess1)
    #ans = brenth(d2_Frg_Y, 1/N/10, 10, args=(phiM, phiS,))
    #ans = y.x    ###IN THIS CASE, WE GET Y FROM PHI--- SO Y IS THE INPUT
    ans = ans.root
    print('phi', phiM, 'l/lb', ans)
    return -1*ans
def findCrit(phiS,guess):
    #bounds = [(guess*.75,guess*1.25)]
    bounds = [(epsilon,1-epsilon)]
    #brentTrip= (guess/2, guess*1.01, guess*4)
    #brentDub =(guess*.9, guess*2)
    #Yc = minimize_scalar(spin_yfromphi,args=(phiS,guess,),bracket=brentTrip,method='Brent')
    #Yc = minimize_scalar(spin_yfromphi,args=(phiS,guess,),bracket=brentTrip,method='Brent',tol=1e-5)

    Yc = minimize(spin_yfromphi, x0=guess, args=(phiS, guess), method='SLSQP', bounds=bounds)
    #Yc = minimize(spin_yfromphi, x0=guess, args=(phiS, guess), method='L-BFGS-B', bounds=bounds)

    phiC = Yc.x
    return phiC, -1*Yc.fun

t1 = time.time()
phiC,Yc = findCrit(phiS, guess=.020014)
# Yc= -1* spin_yfromphi(phiC,phiS,guess=phiC)
print(phiC,Yc,'crit found in ', (time.time()-t1), ' s \n')

def findSpinlow(Y,phiC):
    initial = phiC/2
    bounds = [(epsilon, phiC-epsilon)]
    #result = minimize(FreeEnergyD2reverse, initial, args=(Y,phiS,),method='Powell',bounds=bounds)
    result = minimize(d2_Frg_phiM, initial, args=(Y, phiS,), method='Powell', bounds=bounds)
    #result = fsolve(FreeEnergyD2reverse, x0=initial, args=(Y,phiS))
    return result.x
def findSpinhigh(Y,phiC):
    initial = phiC+ phiC/2
    bounds = [(phiC+epsilon, 1-epsilon)]
    #result = minimize(FreeEnergyD2reverse, initial, args=(Y,phiS),method='Powell',bounds=bounds)
    result = minimize(d2_Frg_phiM, initial, args=(Y, phiS), method='Nelder-Mead', bounds=bounds)
    #result = fsolve(FreeEnergyD2reverse, x0=initial, args=(Y,phiS))
    return result.x
def findSpins(Y,phiC):
    ##THIS FROM LIN
    phiMax = (1-2*phiS)/(1+qc)-epsilon

    phi1 = brenth(d2_Frg_phiM, epsilon, phiC, args=(Y,phiS))
    phi2 = brenth(d2_Frg_phiM, phiC, phiMax,args = (Y,phiS))
    return phi1,phi2

def FBINODAL(variables,Y,phiBulk,spins):

    s1,s2 = spins
    phi1,phi2 = variables
    #print('testing phi1,phi2: ' ,phi1, phi2, 'Y= ', Y, flush=True )
    if math.isnan(phi1) or math.isnan(phi2): return 1e20
    v = (phi2-phiBulk)/(phi2-phi1)
    eqn = v * ftot_rg(phi1, Y, phiS) + (1 - v) * ftot_rg(phi2, Y, phiS)

    ftot = T0* (eqn - ftot_rg(phiBulk, Y, phiS))

    ### OPTION FOR MAKING GAUSSIAN ###
    #ConstRay = np.array([np.float64(s1*.85), np.float64(s2*1.15)])
    #ftot_differentiable = T0* sum((ftot - ConstRay)**2)  ###########TRYING TO MAKE SLSQP WORK

    #return eqn
    return ftot
    #return np.abs(ftot)#_differentiable
def Jac_rgRPA(vars,Y,phiB,spins):
    phi1=vars[0]
    phi2=vars[1]

    if math.isnan(phi1) or math.isnan(phi2):
        print('Phis are Nan')
        return np.empty(2)

    v = (phi2-phiB)/(phi2-phi1)
    x1 = x_solver(phi1,Y)
    x2 = x_solver(phi2,Y)

    f1 = ftot_rg(phi1, Y, phiS, x1)
    f2 = ftot_rg(phi2, Y, phiS, x2)
    df1 = d1_Frg_dphi(phi1, Y, phiS ,x1)
    df2 = d1_Frg_dphi(phi2, Y, phiS ,x2)

    J = np.empty(2)
    J[0] = v*( (f1-f2)/(phi2-phi1) + df1)
    J[1] = (1-v)*( (f1-f2)/(phi2-phi1) + df2)

    return J*T0

def getInitialVsolved(Y,spinlow,spinhigh,phiBulk):
    bounds = [(epsilon,spinlow-epsilon),(spinhigh+epsilon, 1-epsilon)]
    initial_guess=(spinlow*.9, spinhigh*1.10)
    result = minimize(FBINODAL, initial_guess, args=(Y, phiBulk), method='Nelder-Mead', bounds=bounds)#, options = {'fatol': 1e-3, 'xatol': 1e-3})
    #result = minimize(totalFreeEnergyVsolved, initial_guess,args=(Y,),method='Powell',bounds=bounds)
    phi1i,phi2i= result.x
    if phi2i>spinhigh*3:
        return spinlow*.9, spinhigh*1.1

    else:return phi1i,phi2i
def makeconstSLS(Y,phiBulk,s1,s2,l1,l2):
    def seperated(variables):
        return FBINODAL(variables, Y, phiBulk) - ftot_rg(phiBulk, Y, phiS)
    def minPhi1(variables):
        return variables[0] - l1/10
    def minPhi2(variables):
        return variables[1] - s2*1.05 - epsilon
    def maxPhi1(variables):
        return s1*.975 - epsilon - variables[0]
    def maxPhi2(variables):
        return min(l2*2.5,1-epsilon) - variables[1]
    def phi1Lessphi2(variables):
        return variables[1] - variables[0]
    def equPotential(variables):
        return d1_Frg_dphi(variables[0],Y,phiS) - d1_Frg_dphi(variables[1],Y,phiS)

    return [{'type': 'ineq', 'fun': seperated},{'type': 'ineq', 'fun': minPhi1},{'type': 'ineq', 'fun': minPhi2},{'type': 'ineq', 'fun': maxPhi1},{'type': 'ineq', 'fun': maxPhi2},{'type': 'ineq', 'fun': phi1Lessphi2},{'type': 'eq', 'fun': equPotential}]
def min_verify(minObj, Y, phiB, spins):
    s1, s2 = spins
    min1, min2 = min(minObj.x), max(minObj.x)
    mu1,mu2 = d1_Frg_dphi(min1,Y,phiS),d1_Frg_dphi(min2,Y,phiS)
    print(mu1, mu2, '\n\n INITIAL ATTEMPT mu1 and mu2 for ', min1,min2)

    mu_thresh = 20
    redo_flag = 0
    redo_i = 0

    #if 100*np.abs((mu1-mu2)/mu2) < mu_thresh:

    if (min2 < s2*2.7) and min1< (s1 - 10*epsilon) and min2> (s2+10*epsilon)and min1 > 10*epsilon:
            minObj_verified = minObj
            print('that worked')
    else: redo_flag = 1

    while redo_flag==1:
        print('oh shit looks like we have to retry', redo_i)
        redo_i+=1

        initial_guess=(np.float64(s1*.8 - redo_i*.05*s1),np.float64(s2*1.2 + redo_i*.05*s2))
        bounds = [(s1 / 10, s1 - epsilon), (s2 + epsilon, s2*4)]

        minTemp = minimize(FBINODAL, initial_guess, args=(Y, phiB,spins), method='L-BFGS-B', jac=Jac_rgRPA, bounds=bounds)#, options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20, 'maxfun':MINMAX})
        temp_min1, temp_min2 = min(minTemp.x), max(minTemp.x)
        mu1, mu2 = d1_Frg_dphi(temp_min1, Y, phiS), d1_Frg_dphi(temp_min2, Y, phiS)
        print(mu1, mu2, ' mu1 and mu2 for ', temp_min1, temp_min2)

        #if 100 * np.abs((mu1 - mu2) / mu2) < mu_thresh:
        if (min2 < s2 * 2.7) and min1 < (s1 - 10 * epsilon) and min2 > (s2 + 10 * epsilon) and min1> 10*epsilon:
                minObj_verified = minTemp
                print('\nverified')
                redo_flag = 0
        else:
            redo_flag = 1
        if redo_i>5:
            print('too many fails loser')
            return None

    print('this took ', redo_i, 'additional attempts')

    return minObj_verified
def minFtotal(Y,phiC,lastphi1,lastphi2,dy):

    phi1spin,phi2spin = findSpins(Y,phiC)

    print(lastphi1, lastphi2, 'last 1&2')
    print(phi1spin, phi2spin, 'SPINS LEFT/RIGHT')

    #phiB = (phi1spin+phi2spin)/2
    phiB = phiC

    assert np.isfinite(phi1spin), "phi1spin is not a finite number"
    assert np.isfinite(phi2spin), "phi2spin is not a finite number"

    ###USEFUL MAYBE ###

    #phi1i, phi2i = getInitialVsolved(Y, phi1spin, phi2spin,phiB)
    #initial_guess=(np.float64(phi1i),np.float64(phi2i))
    #initial_guess=(np.float64(lastphi1*.9),np.float64(lastphi2*1.1))

    ### GET CONSTRAINTS ###
    const = makeconstSLS(Y,phiB, phi1spin,phi2spin,lastphi1,lastphi2)
    phiMax = (1-2*phiS)/(1 + qc) - epsilon ### FROM LIN ###
    #bounds = [(epsilon, phi1spin - epsilon), (phi2spin+epsilon, 1-epsilon)]
    bounds = [(phi1spin/10, phi1spin - epsilon), (phi2spin+epsilon,  phiMax)]
    t0 = time.time()

    ### MINIMIZER ### DEPENDS ON IF STARTING AT TOP OR NOT ###
    if lastphi1!=phiC:

        ### METHODS TO CHOOSE ### SCIPY.OPTIMIZE.MINIMIZE ###
        M1 = 'TNC'
        M2 = 'L-BFGS-B'
        M3 = 'SLSQP'

        ### MAKE INITIAL ### IN PROGRESS ###
        #initial_guess = (np.float64(lastphi1 * (1 - .04 * (scale_init/.001))), np.float64(lastphi2 * (1 + .04*(scale_init/.001))))
        initial_guess=(np.float64(phi1spin*.6),np.float64(phi2spin*1.48))
        #initial_guess=(np.float64(phi1spin*(1 - Yc/Y - .1)),np.float64(phi2spin*(Yc/Y)+.45))

        print(initial_guess, ' THIS IS INITIAL GUESS FOR Y = ', Y)

        ### DEFINE SPINS AND MINIMIZE ### USES MULTIPLE MINIMIZERS AND CHECKS VALIDITY ###
        spins=[phi1spin,phi2spin]

        maxL1 = minimize(FBINODAL, initial_guess, args=(Y, phiB,spins), method=M1, jac=Jac_rgRPA, bounds=bounds, options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20})# , 'maxfun':MINMAX})
        print(M1, 'M1 minimized, beginning,', M2 ,'M2\n')

        maxL1 = min_verify(maxL1,Y, phiB, spins)

        ### OPTION TO USE LAST FIND AS INITIAL ###
        #initial_guess= (np.float64(min(maxL1.x)*.95),np.float64(max(maxL1.x)*1.1))

        maxL2 = minimize(FBINODAL, initial_guess, args=(Y, phiB,spins), method=M2, jac=Jac_rgRPA, bounds=bounds, options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20, 'maxfun':MINMAX})
        #print(M2, 'M2 minimized, beginning,', M3 ,'M3\n')
        maxL2 = min_verify(maxL2, Y,phiB, spins)

        #initial_guess= (np.float64(phi1spin*.55),np.float64(phi2spin*1.45))
        #maxL3 = minimize(FBINODAL, initial_guess, args=(Y, phiB,spins), method=M3, jac=Jac_rgRPA, bounds=bounds, options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20, 'maxiter':3})
        #print(M3, 'M3 minimized\n')

        ### OPTION TO TOGGLE OFF OTHER MINIMIZERS ###
        #maxL2 = maxL1
        maxL3 = maxL1

        if maxL1 == None: maxL1=maxL2
        elif maxL2 ==None: maxL2 =maxL1

        ### FINDING LOWEST SOLUTION ###
        if(maxL1.fun<=maxL2.fun and maxL1.fun<=maxL3.fun):
            print('M1 was more accurate', maxL1.fun, 'vs.M2/3', maxL2.fun, maxL3.fun)
            phi1min = min(maxL1.x)
            phi2min = max(maxL1.x)
        elif(maxL2.fun<= maxL1.fun and maxL2.fun<=maxL3.fun):
            print('M2 was more accurate', maxL2.fun, 'vs. M1/3', maxL1.fun, maxL3.fun)
            phi1min = min(maxL2.x)
            phi2min = max(maxL2.x)
        else:
            print('M3 was more accurate', maxL3.fun, 'vs.(M1,M2)', maxL1.fun,maxL2.fun)
            phi1min = min(maxL3.x)
            phi2min = max(maxL3.x)
    else:
        ### TOP OF BINODAL GRAPH ### THIS ALWAYS WORKS ###
        initial_guess = (np.float64(phi1spin * .9), np.float64(phi2spin * 1.15))
        spins= [phi1spin,phi2spin]
        maxL1 = minimize(FBINODAL, initial_guess, args=(Y, phiB, spins), method='TNC', jac=Jac_rgRPA, bounds=bounds)#, options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20 })#, 'maxfun':MINMAX})
        maxL1 = min_verify(maxL1, Y,phiB,spins)
        phi1min = min(maxL1.x)
        phi2min = max(maxL1.x)

    ### PRINTING RESULTS FROM MINIMIZER ###
    v = (phi2min-phiB)/(phi2min-phi1min)

    print('\nMINIMIZER COMPLETE FOR Y = ',Y, 'MIN VALUES: phi1,phi2, v = ',phi1min,phi2min,v)
    print('\nThis step took ', time.time()-t0, 's')
    return phi1spin, phi2spin, phi1min,phi2min
def getBinodal(Yc,phiC,minY):
    biphibin= np.array([phiC])
    sphibin = np.array([phiC])

    Ybin = np.array([Yc])
    Ytest= Yc - scale_init

    Y_range= Yc - minY

    while Ytest>minY:
        Y_ratio_done = (Yc -Ytest)/Y_range
        resolution = scale_init *(1 + Y_ratio_done*(scale_final/scale_init))

        #print(Ytest, "until", minY)
        phiLlast,phiDlast = biphibin[0], biphibin[-1]
        biphibin, sphibin = biphibin.flatten(), sphibin.flatten()
        #to avoid dimension bs
        spin1,spin2, phi1,phi2 = minFtotal(Ytest, phiC, phiLlast, phiDlast,resolution)
        print(spin1,spin2, 'these were spins')

        #if phi1<phiLlast and phi2>phiDlast:
        if True:
            phi1=np.array([phi1])
            phi2=np.array([phi2])
            spin1 = np.array([spin1])
            spin2 = np.array([spin2])
            biphibin = np.concatenate((phi1, biphibin, phi2))
            sphibin = np.concatenate((spin1, sphibin, spin2))
            Ybin = np.concatenate(([Ytest], Ybin, [Ytest]))

        else: print('someglitch, repeating with a skipped step')
        ####HIGHER RESOLUTION AT TOP OF PHASE DIAGRAM###################
        #resolution = scale_init * np.exp((Yc / Ytest) ** 3) / np.exp(1)
        print("\nNEXT YTEST CHANGED BY:", resolution, "and Ytest=", Ytest)
        Ytest-=resolution

    return sphibin, biphibin, Ybin
