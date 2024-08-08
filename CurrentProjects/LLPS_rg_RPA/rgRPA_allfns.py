import numpy as np
from OldProteinProjects.SCDcalc import *
from scipy import integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from rgRPA_init import *
from scipy.optimize import brenth, brent
import math
import time

########################ConstANTS################################
T0=100
iterlim=150
qL = np.array(qs)
Q = np.sum(qL*qL)/N

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
    xeesum=0
    for i in range(0,N-1):
        xeesum += sigS[i]*np.exp((-1/6)*x*(k*k)*i)
    return xeesum/N
def xee_r(k,x,sigS):
    xeesum=0
    for i in range(0,N-1):
        xeesum += sigS[i]*i*i*np.exp((-1/6)*x*(k*k)*i)
    return xeesum/N
def gk(k,x):
    gksum=0
    for i in range(0,N-1):
        gksum+= sigShift_gk[i]*np.exp((-1/6)*x*k*k*i)
    return gksum/N
def gk_r(k,x):
    gksum=0
    for i in range(0,N-1):
        gksum+= sigShift_gk[i]*i*i*np.exp((-1/6)*x*k*k*i)
    return gksum/N
def ck(k,x,sigs):
    cksum=0
    for i in range(0,N-1):
        cksum += sigs[i]*np.exp((-1/6)*x*k*k*i)
    return cksum/N
def ck_r(k,x,sigs):
    cksum=0
    for i in range(0,N-1):
        cksum += i*i*sigs[i]*np.exp((-1/6)*x*k*k*i)
    return cksum/N

#d1
def d1_xee(k,x,sigS):
    xeesum=0
    for i in range(0,N-1):
        xeesum += sigS[i]*np.exp((-1/6)*x*(k*k)*i) *(-1/6)*(k*k)*i
    return xeesum/N
def d1_xee_r(k,x,sigS):
    xeesum=0
    for i in range(0,N-1):
        xeesum += sigS[i]*i*i*np.exp((-1/6)*x*(k*k)*i) *(-1/6)*(k*k)*i
    return xeesum/N
def d1_gk(k,x):
    gksum=0
    for i in range(0,N-1):
        gksum+= sigShift_gk[i]*np.exp((-1/6)*x*k*k*i) *(-1/6)*(k*k)*i
    return gksum/N
def d1_gk_r(k,x):
    gksum=0
    for i in range(0,N-1):
        gksum+= sigShift_gk[i]*i*i*np.exp((-1/6)*x*k*k*i)*(-1/6)*(k*k)*i
    return gksum/N
def d1_ck(k,x,sigs):
    cksum=0
    for i in range(0,N-1):
        cksum+= sigs[i]*np.exp((-1/6)*x*k*k*i)*(-1/6)*(k*k)*i
    return cksum/N
def d1_ck_r(k,x,sigs):
    cksum=0
    for i in range(0,N-1):
        cksum+= i*i*sigs[i]*np.exp((-1/6)*x*k*k*i)*(-1/6)*(k*k)*i
    return cksum/N

#d2
def d2_xee(k,x,sigS):
    xeesum=0
    for i in range(0,N-1):
        xeesum += sigS[i]*np.exp((-1/6)*x*(k*k)*i) *(1/36)*(k*k)*i*(k*k)*i
    return xeesum/N
def d2_xee_r(k,x,sigS):
    xeesum=0
    for i in range(0,N-1):
        xeesum += sigS[i]*i*i*np.exp((-1/6)*x*(k*k)*i)*(1/36)*(k*k)*i*(k*k)*i
    return xeesum/N
def d2_gk(k,x):
    gksum=0
    for i in range(0,N-1):
        gksum+= sigShift_gk[i]*np.exp((-1/6)*x*k*k*i)*(1/36)*(k*k)*i*(k*k)*i
    return gksum/N
def d2_gk_r(k,x):
    gksum=0
    for i in range(0,N-1):
        gksum+= sigShift_gk[i]*i*i*np.exp((-1/6)*x*k*k*i)*(1/36)*(k*k)*i*(k*k)*i
    return gksum/N
def d2_ck(k,x,sigs):
    cksum=0
    for i in range(0,N-1):
        cksum+= sigs[i]*np.exp((-1/6)*x*k*k*i)*(1/36)*(k*k)*i*(k*k)*i
    return cksum/N
def d2_ck_r(k,x,sigs):
    cksum=0
    for i in range(0,N-1):
        cksum+= i*i*sigs[i]*np.exp((-1/6)*x*k*k*i)*(1/36)*(k*k)*i*(k*k)*i
    return cksum/N

############SOLVING FOR X #####################################################
def x_solver(phiM,Y):
    #print(x_eqn(1/(N*1000),phiM,Y),'lowbound', x_eqn(100*N,phiM,Y), 'upper')

    sln = brenth(x_eqn, 1/10/N, 100*N, args=(phiM,Y))
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

    ionConst = k * k * Y / (4 * np.pi) + phiS + qc*phiM
    v = 4*np.pi/3
    r12 = ex*g - c*c
    phi2 = phiM*phiM
    m12 = g*exr + ex*gkr - 2*c*ckr

    c0 = phi2*(r12) + phiM*(ex/v + g*(ionConst)) + (ionConst)/v
    d0 = (exr/v + gkr*(ionConst) + phiM*(m12))
    f0 = phi2*(g*d1ex + d1g*ex - 2*c*d1c) + phiM*d1ex/v + phiM*(ionConst)*d1g
    h0 = d1exr/v + d1gkr*(ionConst) + phiM*(gkr*d1ex + exr*d1g - 2*ckr*d1c + g*d1exr + ex*d1gkr - 2*c*d1ckr) ###LAST TERM HERE LOOKS WRONG::: FIXED ckr --> d1ckr

    return k * k * k * k * ((h0*c0*c0 - d0*f0*c0) / (c0 * c0 * c0)) * (1 / (2 * np.pi * np.pi))

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
def rgFP(phiM, Y, phiS,x=None):
    x = x_solver(phiM, Y) if x==None else x

    def rgFPint(k,Y,phiM,phiS):
        xe = xee(k, x, sigS=sigShift_xe)
        g = gk(k, x)
        c = ck(k, x, sigs=sigShift_ck)
        v = (4*np.pi/3)*np.exp(-1*k*k/6)
        nu = k*k*Y/4/np.pi + qc*phiM + phiS
        N1 = nu + phiM*(g*nu*v + xe) +v*phiM*phiM*(g*xe-c*c)
        A = N1/nu
        B= 1+ Q/nu
        return (k*k/4/np.pi/np.pi)*np.log(A/B)
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(rgFPint, lowerlim, upperlim, args=(Y, phiM, phiS), limit=iterlim)
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

################FIRST DERIVATIVES##############################
def d1_Frg_dphi(phiM,Y,phiS,x=None, dx = None):
    x = x_solver(phiM,Y) if x==None else x
    dx = d1_x_solver(phiM,Y,x) if dx ==None else dx
    phic = qc*phiM
    phiW = 1 - phiM - phic - phiS

    def ds_1comp(x):
        ##ALSO FRM LIN FOR SPEED AND NO ERRORS
        return (np.log(x + 1e5 * (x <= 0)) + 1) * (x > 0)

    ds_dphi = (ds_1comp(phiM)/N + qc*ds_1comp(phic) - (1+qc)*ds_1comp(phiW))*(phiM>0)
    dfion_dphi= (-1*np.sqrt(np.pi)*qc*np.sqrt((phic + phiS)/Y))/(Y*(2*np.sqrt(np.pi)*np.sqrt((phic)/Y)+1))

    def dfpintegrand(k,Y,phiM,phiS):
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
    upper,lower = 0,np.inf
    result = integrate.quad(dfpintegrand, lower, upper, args=(Y,phiM,phiS,), limit=iterlim)
    dfp_dphi = result[0] / (4 * np.pi * np.pi)

    return ds_dphi+dfion_dphi+dfp_dphi + 4*np.pi*phiM/3 #d1f0

#####################SECOND DERIVATIVE FREE ENERGIES 2 VERSIONS#############################
def d2_Frg_Y(Y,phiM,phiS,x = None, dx = None, ddx = None):
    x = x_solver(phiM, Y) if x == None else x
    dx = d1_x_solver(phiM, Y,x) if dx == None else dx
    ddx = d2_x_solver(phiM, Y,x,dx) if ddx == None else ddx
    phic = qc*phiM
    phiW = 1- phiM - phic - phiS
    def d2s_1comp(x):
        return (x>0)/(x + (x==0))
    #################Entropyd2##########
    d2s = (d2s_1comp(phiM)/N + qc*qc*d2s_1comp(phic) + (1+qc)*(1+qc)*d2s_1comp(phiW))*(phiM>0)

    #####d2Fion#################
    c = 4*np.pi/Y
    rho = phic + phiS

    k = np.sqrt((c)*(rho))
    dk = (qc/2)*np.sqrt(c/rho)
    ddk = (-1*qc*qc/4)*np.sqrt(c/(rho*rho*rho))

    d2fion = (-1/(4*np.pi))*(k*(k*(k+1)*ddk + (k+2)*dk*dk))/((k+1)*(k+1))

    #d2f0
    d2f0 = 4*np.pi/3

    #################Electrofreeenergyd2###########
    def d2_FP_toint(k, Y, phiM):
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

        return k2 * (Num1 / Den - (Num2 * Num2) / (Den * Den))

    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(d2_FP_toint, lowerlim, upperlim, args=(Y, phiM,), limit=iterlim)

    d2fp = result[0] / (4 * np.pi * np.pi)
    #print(d2s+d2f0,d2fp,d2fion, Y)
    return (d2s + d2fp + d2fion +d2f0)
def d2_Frg_phiM(phiM,Y,phiS,x=None,dx=None,ddx=None):
    x = x_solver(phiM, Y) if x == None else x
    dx = d1_x_solver(phiM, Y, x) if dx == None else dx
    ddx = d2_x_solver(phiM, Y, x, dx) if ddx == None else ddx
    phic = qc * phiM
    phiW = 1 - phiM - phic - phiS

    def d2s_1comp(x):
        return (x > 0) / (x + (x == 0))

    #################Entropyd2##########
    d2s = (d2s_1comp(phiM) / N + qc * qc * d2s_1comp(phic) + (1 + qc) * (1 + qc) * d2s_1comp(phiW)) * (phiM > 0)

    #####d2Fion#################
    c = 4 * np.pi / Y
    rho = qc * phiM + phiS

    k = np.sqrt((c) * (rho))
    dk = (qc / 2) * np.sqrt(c / rho)
    ddk = (-1 * qc * qc / 4) * np.sqrt(c / (rho * rho * rho))

    d2fion = (-1 / (4 * np.pi)) * (k * (k * (k + 1) * ddk + (k + 2) * dk * dk)) / ((k + 1) * (k + 1))

    # d2f0
    d2f0 = 4 * np.pi / 3

    #################Electrofreeenergyd2###########
    def d2_FP_toint(k, Y, phiM):
        xe = xee(k, x, sigS=sigShift_xe)
        d1xe = d1_xee(k, x, sigS=sigShift_xe)
        d2xe = d2_xee(k, x, sigS=sigShift_xe)
        g = gk(k, x)
        d1g = d1_gk(k, x)
        d2g = d2_gk(k, x)
        c = ck(k, x, sigs=sigShift_ck)
        d1c = d1_ck(k, x, sigs=sigShift_ck)
        d2c = d2_ck(k, x, sigs=sigShift_ck)

        rho = k * k * Y / (4 * np.pi) + phiS + qc * phiM
        dx2, qc2, phi2, rho2 = dx * dx, qc * qc, phiM * phiM, rho * rho

        v2 = (4 * np.pi / 3) * np.exp(-1 * k * k / 6)
        D2BIG = g * d2xe * dx2 + g * d1xe * ddx + 2 * d1xe * d1g * dx2 + xe * d2g * dx2 + xe * d1g * ddx - 2 * c * d2c * dx2 - 2 * c * d1c * ddx - 2 * d1c * d1c * dx2
        D = xe * g - c * c
        DD = g * d1xe * dx + xe * d1g * dx - 2 * c * d1c * dx

        Num1 = (-2 * qc * v2 * phi2 * (DD) / (
            rho2) + 4 * v2 * phiM * (
                    DD) / rho + 2 * d1xe * dx / rho + phiM * (
                        d2xe * dx2 / rho + d1xe * ddx / rho - 2 * qc * d1xe * dx / (
                    rho2) + 2 * qc2 * xe / (rho * rho2) + v2 * (
                                d2g * dx2 + d1g * ddx)) + v2 * phi2 * (
                    D2BIG) / rho + 2 * qc2 * v2 * phi2 * (D) / (
                        rho * rho2) - 4 * qc * v2 * phiM * (D) / (
                    rho2) + 2 * v2 * D / rho - 2 * qc * xe / (
                    rho2) + 2 * v2 * d1g * dx)

        Den = (v2 * phi2 * D / rho + phiM * (xe / rho + v2 * g) + 1)

        Num2 = (v2 * phi2 * (DD) / rho + phiM * (
                d1xe * dx / rho - qc * xe / (
            rho2) + v2 * d1g * dx) - qc * v2 * phi2 * D / (
                    rho2) + 2 * v2 * phiM * D / rho + xe / rho + v2 * g)
        return k * k * (Num1 / Den - (Num2 * Num2) / (Den * Den))

    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(d2_FP_toint, lowerlim, upperlim, args=(Y, phiM,), limit=iterlim)

    d2fp = result[0] / (4 * np.pi * np.pi)
    # print(d2s+d2f0,d2fp,d2fion, Y)
    return (d2s + d2fp + d2fion + d2f0)

##########################################################################
#######SOLVER METHODS BELOW###############################################
##########################################################################

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
    y = fsolve(d2_Frg_Y, args=(phiM, phiS,), x0=guess1)
    #y = brenth(d2_Frg_Y, 1/N/10, 10, args=(phiM, phiS,))
    print('phi', phiM, 'l/lb', y)
    return -1*y
def findCrit(phiS,guess):

    bounds = [(.005,.05)]

    ###FROM LIN
    phi_max = (1 - 2 * phiS) / (1 + qc)
    ini1, ini3, ini2 = 1e-6, 1e-2, phi_max*2/3
    Yc = minimize(spin_yfromphi, x0=guess, args=(phiS, guess,), method='L-BFGS-B', bounds=bounds)
    #result = brent(spin_yfromphi, args=(phiS,guess), brack = (ini1,ini3,ini2), full_output=1)
    #phiC,Yc = result[0], result[1]
    phiC = Yc.x
    #phiC = Yc
    return phiC, -1*Yc.fun
    #return phiC, Yc

t1 = time.time()
phiC,Yc = findCrit(phiS, guess=.02416)
t2 = time.time()
# phiC =.019875
# Yc= -1* spin_yfromphi(phiC,phiS,guess=phiC)
print(phiC,Yc,'crit found in ', (t2-t1), ' s \n')

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

def FBINODAL(variables,Y,phiBulk):
    phi1,phi2 = variables
    print('testing phi1,phi2: ' ,phi1, phi2)
    if math.isnan(phi1) or math.isnan(phi2): return 1e20
    v = (phi2-phiBulk)/(phi2-phi1)
    eqn = v * ftot_rg(phi1, Y, phiS) + (1 - v) * ftot_rg(phi2, Y, phiS)
    #return eqn
    return T0*(eqn - ftot_rg(phiBulk, Y, phiS))
def getInitialVsolved(Y,spinlow,spinhigh,phiBulk):
    bounds = [(epsilon,spinlow-epsilon),(spinhigh+epsilon, 1-epsilon)]
    initial_guess=(spinlow*.899, spinhigh*1.101)
    result = minimize(FBINODAL, initial_guess, args=(Y, phiBulk), method='Nelder-Mead', bounds=bounds)
    #result = minimize(totalFreeEnergyVsolved, initial_guess,args=(Y,),method='Powell',bounds=bounds)
    phi1i,phi2i= result.x
    return phi1i,phi2i
def makeconstSLS(Y,phiBulk):
    def seperated(variables):
        return FBINODAL(variables, Y, phiBulk) - ftot_rg(phiC, Y, phiS)
    def minPhi1(variables):
        return variables[0]
    def minPhi2(variables):
        return variables[1]
    def maxPhi1(variables):
        return 1-variables[0]
    def maxPhi2(variables):
        return 1-variables[1]

    return [{'type': 'ineq', 'fun': seperated},{'type': 'ineq', 'fun': minPhi1},{'type': 'ineq', 'fun': minPhi2},{'type': 'ineq', 'fun': maxPhi1},{'type': 'ineq', 'fun': maxPhi2}]
def minFtotal(Y,phiC,lastphi1,lastphi2):

    # phi1spin = findSpinlow(Y, phiC)[0]
    # phi2spin = findSpinhigh(Y, phiC)[0]
    phi1spin,phi2spin = findSpins(Y,phiC)
    print(lastphi1, lastphi2, 'last 1&2')
    print(phi1spin, phi2spin, 'SPINS LEFT/RIGHT')
    phiB = (phi1spin+phi2spin)/2
    assert np.isfinite(phi1spin), "phi1spin is not a finite number"
    assert np.isfinite(phi2spin), "phi2spin is not a finite number"

    #phi1i, phi2i = getInitialVsolved(Y, phi1spin, phi2spin,phiB)
    #if lastphi1==phiC:
    initial_guess=(np.float64(phi1spin*.9),np.float64(phi2spin*1.1))
    #else: initial_guess=(lastphi1*.98, lastphi2*1.03)
    #initial_guess=(phi1spin*.9,phi2spin*1.1)
    #initial_guess=(phi1spin*.899,phi2spin*1.301)
    #phi2Max = (1-2*phiS)/(1+qc)#########FROM LIN CODE GITHUB ????
    print(initial_guess)
    const = makeconstSLS(Y,phiB)
    #phiMax = (1-2*phiS)/(1 + qc) - epsilon
    bounds = [(epsilon, phi1spin - epsilon), (phi2spin+epsilon, 1-epsilon)]
    maxL = minimize(FBINODAL, initial_guess, args=(Y, phiB), method='L-BFGS-B', jac=Jac_fgRPA , bounds=bounds, options={'ftol':1e-11, 'gtol':1e-11, 'eps':1e-11})
    #maxL = minimize(FBINODAL,initial_guess,args=(Y,phiB),method='SLSQP',jac=Jac_fgRPA)#,bounds=bounds,constraints=const)#, options={'ftol':1e-10, 'gtol':1e-10, 'eps':1e-8} )
    #maxL = minimize(FBINODAL,initial_guess,args=(Y,phiB),method='CG',jac=Jac_fgRPA,bounds=bounds)#,options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20} )

    #maxL = minimize(FBINODAL, initial_guess, args=(Y,phiB), method='SLSQP', constraints=const,bounds=bounds)
    #maxL = minimize(FBINODAL, initial_guess, args=(Y,phiB), method='Powell', bounds=bounds)
    #maxL = minimize(FBINODAL, initial_guess, args=(Y,phiB), method='Nelder-Mead', bounds=bounds)
    #maxparams = maxL.x
    #phi1min,phi2min = maxparams
    phi1min = min(maxL.x)
    phi2min = max(maxL.x)
    v = (phi2min-phiB)/(phi2min-phi1min)

    print('\nwe have finished minimizing for Y = ',Y, 'just cuz curious: phi1,phi2, v = ',phi1min,phi2min,v)

    return phi1spin, phi2spin, phi1min,phi2min
def Jac_fgRPA(vars,Y,phiB):
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
def getBinodal(Yc,phiC,minY):
    biphibin= np.array([phiC])
    sphibin = np.array([phiC])
    Ybin = np.array([Yc])
    Ytest= Yc - scale_init

    Y_range= Yc - minY

    while Ytest>minY:
        Y_ratio_done = (Yc -Ytest)/Y_range
        #print(Ytest, "until", minY)
        phiLlast,phiDlast = biphibin[0], biphibin[-1]
        spin1,spin2, phi1,phi2 = minFtotal(Ytest, phiC, phiLlast, phiDlast)
        print(spin1,spin2)

        if phi1<phiLlast and phi2>phiDlast:
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
        resolution = scale_init *(1 + Y_ratio_done*(scale_final/scale_init))
        print("NEXT YTEST CHANGED BY:", resolution, "and Ytest=", Ytest)
        Ytest-=resolution

    return sphibin, biphibin, Ybin
