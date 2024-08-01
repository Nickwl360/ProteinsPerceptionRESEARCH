import numpy as np
from OldProteinProjects.SCDcalc import *
from scipy import integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from rgRPA_init import *
from scipy.optimize import brenth, brent

########################ConstANTS################################
T0=1e5
iterlim=200

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
#############SEQUENCE SPECIFIC CHARGE###########################

#FROM LIN TO TEST
# mel = np.kron(qs, qs).reshape((N, N))
# Tel = np.array([ np.sum(mel.diagonal(n) + mel.diagonal(-n)) \
#                     for n in range(N)])
# Tel[0] /= 2
#
# Tex = 2*np.arange(N,0,-1)
# Tex[0] /= 2

##############STRUCTURE FACTORS & DERIVATIVES###########################
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
        xeesum += sigS[i]*np.exp((-1/6)*x*(k*k)*i) *(-1/6)*(k*k)*i*(-1/6)*(k*k)*i
    return xeesum/N
def d2_xee_r(k,x,sigS):
    xeesum=0
    for i in range(0,N-1):
        xeesum += sigS[i]*i*i*np.exp((-1/6)*x*(k*k)*i)*(-1/6)*(k*k)*i*(-1/6)*(k*k)*i
    return xeesum/N
def d2_gk(k,x):
    gksum=0
    for i in range(0,N-1):
        gksum+= sigShift_gk[i]*np.exp((-1/6)*x*k*k*i)*(-1/6)*(k*k)*i*(-1/6)*(k*k)*i
    return gksum/N
def d2_gk_r(k,x):
    gksum=0
    for i in range(0,N-1):
        gksum+= sigShift_gk[i]*i*i*np.exp((-1/6)*x*k*k*i)*(-1/6)*(k*k)*i*(-1/6)*(k*k)*i
    return gksum/N
def d2_ck(k,x,sigs):
    cksum=0
    for i in range(0,N-1):
        cksum+= sigs[i]*np.exp((-1/6)*x*k*k*i)*(-1/6)*(k*k)*i*(-1/6)*(k*k)*i
    return cksum/N
def d2_ck_r(k,x,sigs):
    cksum=0
    for i in range(0,N-1):
        cksum+= i*i*sigs[i]*np.exp((-1/6)*x*k*k*i)*(-1/6)*(k*k)*i*(-1/6)*(k*k)*i
    return cksum/N

############SOLVING FOR X #####################################################
def x_solver(phiM,Y):
    #print(x_eqn(1/(N*1000),phiM,Y),'lowbound', x_eqn(100*N,phiM,Y), 'upper')
    sln = brenth(x_eqn, 1/100/N, 1000*N, args=(phiM,Y))
    #sln = fsolve(x_eqn,np.array([.5]),args=(phiM,Y,))
    #sln = root_scalar(x_eqn,x0=.5, args=(phiM,Y,))
    #bounds=[(0,1)]
    #sln = minimize(x_eqn, np.array([.5]),args=(phiM,Y,),method='Nelder-Mead',bounds=bounds)
    return sln
def x_eqn(x,phiM,Y):
    eqn = 1 - 1/x - (N/(18*(N-1))) * integrate.quad(x_eqn_toint,epsilon,np.inf,args=(phiM,Y,x),limit=iterlim)[0]
    return T0*eqn
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

    return Nconst*I1/((1/(x*x))-Nconst*I2)
def d1_x_eqn_I1int(k,phiM,Y,x):
    exr = xee_r(k, x, sigS=sigShift_xe)
    ex = xee(k,x,sigS=sigShift_xe)
    gkr = gk_r(k,x)
    g = gk(k,x)
    c = ck(k,x,sigs=sigShift_ck)
    ckr = ck_r(k,x,sigs= sigShift_ck)

    ionConst = k*k*Y/(4*np.pi) + phiS

    b0 = ex*gkr + g*exr - 2*c*ckr + qc*gkr
    c0 = phiM*(g*(ionConst+qc*phiM)+3*ex/(4*np.pi)) + (3/(4*np.pi))*(ionConst+qc*phiM) + phiM*phiM*(ex*g-c*c)
    e0 = phiM*qc*g + g*(ionConst+qc*phiM) + 2*phiM*(ex*g-c*c) + (3/(4*np.pi))*(qc+ex)
    f0 = (gkr*(ionConst+qc*phiM)+phiM*(ex*gkr+g*exr-2*c*ckr)+3*exr/(4*np.pi))

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

    ionConst = k*k*Y/(4*np.pi) + phiS
    a0 = d1gkr*(ionConst+qc*phiM)+phiM*(gkr*d1ex+ex*d1gkr+exr*d1g+g*d1exr-2*ckr*d1c-2*c*d1ckr)+3*d1exr/(4*np.pi)
    c0 = phiM*(g*(ionConst+qc*phiM)+3*ex/(4*np.pi)) + (3/(4*np.pi))*(ionConst+qc*phiM) + phiM*phiM*(ex*g-c*c)
    d0 = phiM*d1g*(ionConst+qc*phiM) + phiM*3*d1ex/(4*np.pi) + phiM*phiM*(g*d1ex+ ex*d1g-2*c*d1c)
    f0 = (gkr*(ionConst+qc*phiM)+phiM*(ex*gkr+g*exr-2*c*ckr)+3*exr/(4*np.pi))

    return k*k*k*k*((a0*c0 - d0*f0)/(c0*c0))*(1/(2*np.pi*np.pi))

#####MY VERSION
#d2
def d2_x_solver(phiM,Y,x=None,dx=None):
    x=x_solver(phiM,Y) if x==None else x
    dx=d1_x_solver(phiM,Y ,x) if dx==None else dx

    Nconst = N/(18*(N-1))
    lower,upper=0,np.inf
    I1,I2 = integrate.quad(d2_x_eqn_I1int,lower,upper, args=(phiM,Y,x,dx,),limit=iterlim)[0] ,integrate.quad(d2_x_eqn_I2int,lower,upper, args=(phiM,Y,x,dx,),limit=iterlim)[0]

    return (Nconst*I1 + 2*dx*dx/(x*x*x))/((1/(x*x)) - Nconst*I2)
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

    ionConst = k * k * Y / (4 * np.pi) + phiS + qc*phiM
    ##########DOUBLE CHECK THIS TERM!!!!!!!!!!!!!!
    a0 = (phiM*phiM*(g*d1ex*dx + ex*d1g*dx - 2*c*d1c*dx) + 2*phiM*(ex*g-c*c) + phiM*(qc*g + 3*d1ex*dx/(4*np.pi) +d1g*dx*(ionConst)) + (3/(4*np.pi))*(qc+ex) + g*(ionConst))#
    b0=  (g*exr + qc*gkr + ex*gkr -2*c*ckr + 3*d1exr*dx/(np.pi*4)+ d1gkr*dx*(ionConst)+phiM*(gkr*d1ex*dx+exr*d1g*dx-2*ckr*d1c*dx+g*d1exr*dx+ex*d1gkr*dx-2*c*d1ckr*dx))
    c0 = phiM*phiM*(ex*g-c*c)+phiM*(3*ex/(4*np.pi)+g*(ionConst))+(3/(4*np.pi))*(ionConst)
    d0 = (3*exr/(4*np.pi)+gkr*(ionConst) + phiM*(g*exr+ex*gkr-2*c*ckr))
    e0 = (phiM*phiM*(g*d1ex*dx + ex*d1g*dx - 2*c*d1c*dx)+ 2*phiM*(ex*g-c*c) + phiM*(qc*g +3*d1ex*dx/(4*np.pi) + d1g*dx*(ionConst))+3*(qc+ex)/(4*np.pi)+g*(ionConst))

    #f0 = phiM*phiM*g*d1ex + phiM*phiM*d1g*ex - 2*c*d1c*phiM*phiM - phiM*3*d1ex/(4*np.pi) - phiM*(ionConst+qc*phiM)*d1g
    g0 = (phiM*phiM*(2*d1ex*d1g*dx*dx - 2*d1c*d1c*dx*dx + g*d2ex*dx*dx + ex*d2g*dx*dx - 2*c*d2c*dx*dx) + 4*phiM*(g*d1ex*dx + ex*d1g*dx - 2*c*d1c*dx) + phiM*(2*qc*d1g*dx + 3*d2ex*dx*dx/(4*np.pi) + d2g*dx*dx*(ionConst)) + 2*qc*g + 2*(ex*g-c*c) + 3*d1ex*dx/(2*np.pi) + 2*d1g*dx*(ionConst))
    #h0 = 3*ex/(4*np.pi)+d1gkr*(ionConst+qc*phiM) +phiM*(gkr*d1ex+exr*d1g-2*ckr*d1c+g*d1exr+ex*d1gkr-2*c*ckr)
    i0 = 2*qc*d1gkr*dx + 2*(gkr*d1ex*dx + exr*d1g*dx - 2*ckr*d1c*dx + g*d1exr*dx + ex*d1gkr*dx - 2*c*d1ckr*dx) + 3*d2exr*dx*dx/(4*np.pi) +d2gkr*dx*dx*(ionConst) + phiM*(2*d1g*d1exr*dx*dx + 2*d1ex*d1gkr*dx*dx - 4*d1c*d1ckr*dx*dx + gkr*d2ex*dx*dx + exr*d2g*dx*dx - 2*ckr*d2c*dx*dx + g*d2exr*dx*dx + ex*d2gkr*dx*dx - 2*c*d2ckr*dx*dx)

    return k*k*k*k*((2*d0*e0*e0 - d0*g0*c0 - 2*a0*b0*c0 + i0*c0*c0)/(c0*c0*c0))*(1/(2*np.pi*np.pi))
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
    #d1ckr = d1_ck_r(k,x,sigs = sigShift_ck)

    ionConst = k * k * Y / (4 * np.pi) + phiS + qc*phiM


    #a0 = (phiM*phiM*(g*d1ex*dx + ex*d1g*dx - 2*c*d1c*dx) + 2*phiM*(ex*g-c*c) + phiM*(qc*g + 3*d1ex*dx/(4*np.pi) +d1g*dx*(ionConst+qc*phiM)) + (3/(4*np.pi))*(qc+ex) + g*(ionConst+qc*phiM))#
    #b0=  (g*exr + qc*gkr + ex*gkr -2*c*ckr + 3*d1exr*dx/(np.pi*4)+ d1gkr*dx*(ionConst+qc*phiM)+phiM*(gkr*d1ex*dx+exr*d1g*dx-2*ckr*d1c*dx+g*d1exr*dx+ex*d1gkr*dx-2*c*d1ckr*dx))
    c0 = phiM*phiM*(ex*g-c*c)+phiM*(3*ex/(4*np.pi)+g*(ionConst))+(3/(4*np.pi))*(ionConst)
    d0 = (3*exr/(4*np.pi)+gkr*(ionConst) + phiM*(g*exr+ex*gkr-2*c*ckr))
    #e0 = (phiM*phiM*(g*d1ex*dx + ex*d1g*dx - 2*c*d1c*dx)+ 2*phiM*(ex*g-c*c) + phiM*(qc*g +3*d1ex*dx/(4*np.pi) + d1g*dx*(ionConst+qc*phiM))+3*(qc+ex)/(4*np.pi)+g*(ionConst+qc*phiM))

    f0 = phiM*phiM*g*d1ex + phiM*phiM*d1g*ex - 2*c*d1c*phiM*phiM + phiM*3*d1ex/(4*np.pi) + phiM*(ionConst)*d1g
    #g0 = (phiM*phiM*(2*d1ex*d1g*dx*dx - 2*d1c*d1c*dx*dx + g*d2ex*dx*dx + ex*d2g*dx*dx - 2*c*d2c*dx*dx) - 4*phiM*(g*d1ex*dx + ex*d1g*dx - 2*c*d1c*dx) - phiM*(2*qc*d1g*dx + 3*d2ex*dx*dx/(4*np.pi) + d2g*dx*dx*(ionConst+qc*phiM)) - 2*qc*g - 2*(ex*g-c*c) - 3*d1ex*dx/(2*np.pi) - 2*d1g*dx*(ionConst+qc*phiM))
    h0 = 3*d1exr/(4*np.pi) + d1gkr*(ionConst) + phiM*(gkr*d1ex + exr*d1g - 2*ckr*d1c + g*d1exr + ex*d1gkr - 2*c*ckr)
    #i0 = 2*qc*d1gkr*dx+2*(gkr*d1ex*dx+exr*d1g*dx-2*ckr*d1c*dx+g*d1exr*dx+ex*d1gkr*dx-2*c*d1ckr*dx)+3*d2ex*dx*dx/(4*np.pi) +d2gkr*dx*dx*(ionConst+qc*phiM)+phiM*(2*d1g*d1exr*dx*dx+2*d1ex*d1gkr*dx*dx-4*d1c*d1ckr*dx*dx+gkr*d2ex*dx*dx+exr*d2g*dx*dx-2*ckr*d2c*dx*dx+g*d2exr*dx*dx+ex*d2gkr*dx*dx-2*c*d2ckr*dx*dx)

    return k * k * k * k * ((h0*c0*c0 - d0*f0*c0) / (c0 * c0 * c0)) * (1 / (2 * np.pi * np.pi))

x = x_solver(.3,.1)
dx = d1_x_solver(.3,.1,x)
ddx = d2_x_solver(.3,.1,x,dx)
print(x,dx,ddx)

#################FREE ENERGIES#####################################
def ftot_gaussIons(phiM, Y, phiS):
    return entropy(phiM, phiS) + fion(phiM, Y, phiS) + rgFP(phiM, Y, phiS)
def rgFP(phiM, Y, phiS,x=None):
    x = x_solver(phiM, Y) if x==None else x

    def rgFPintegrand(k, Y, phiM, phiS):
        xe = xee(k, x, sigS=sigShift_xe)
        g = gk(k, x)
        c = ck(k, x, sigs=sigShift_ck)
        v2 = (4 * np.pi / 3) * np.exp((-1 / 6) * k * k)
        vc = k * k * Y / (4 * np.pi) + phiS

        return k * k * np.log(1 + phiM * (xe / (vc + qc * phiM) + v2 * g) + phiM * phiM * (v2 / vc) * (xe * g - c * c))

    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(rgFPintegrand, lowerlim, upperlim, args=(Y, phiM, phiS), limit=iterlim)
    fp = result[0] / (4 * np.pi * np.pi)
    return fp
def fion(phiM, Y, phiS):
    kl = np.sqrt(4 * np.pi * (phiS + qc * phiM) / Y)
    return (-1 / (4 * np.pi)) * (np.log(1 + kl) - kl + .5 * kl * kl)
def entropy(phiM, phiS):
    phiC = qc * phiM
    phiW = 1 - phiM - phiS - phiC
    #################FIGURE OUT LOGIC FOR 0s
    if phiS != 0:
        return (phiM / N) * np.log(phiM) + phiS * np.log(phiS) + phiC * np.log(phiC) + phiW * np.log(phiW)
    else:
        return (phiM / N) * np.log(phiM) + phiC * np.log(phiC) + phiW * np.log(phiW)

################FIRST DERIVATIVES##############################
def d1ftotGauss_dphi(phiM,Y,phiS,x=None, dx = None):
    x = x_solver(phiM,Y) if x==None else x
    dx = d1_x_solver(phiM,Y) if dx ==None else dx
    ds_dphi =np.log(phiM)/N + 1/N - 1 + qc*np.log(qc*phiM) + (-1*qc-1)*np.log(1-qc*phiM -phiS - phiM)
    dfion_dphi= (-1*np.sqrt(np.pi)*qc*np.sqrt((qc*phiM+phiS)/Y))/(Y*(2*np.sqrt(np.pi)*np.sqrt((qc*phiM)/Y)+1))
    def dfpintegrand(k,Y,phiM,phiS):
        xe = xee(k,x, sigS=sigShift_xe)
        d1xe = d1_xee(k,x,sigS=sigShift_xe)
        g = gk(k,x)
        d1g = d1_gk(k,x)
        c = ck(k,x,sigs=sigShift_ck)
        d1c = d1_ck(k,x,sigs=sigShift_ck)
        ionConst = k*k*Y/(4*np.pi) + phiS + qc*phiM
        v2 = (4*np.pi/3)*np.exp(-k*k/6)
        num = (v2*phiM*phiM*(g*d1xe*dx + xe*d1g*dx - 2*c*d1c*dx)/(ionConst)+ phiM*(d1xe*dx/ionConst- qc*xe/(ionConst*ionConst) + v2*d1g*dx)- qc*v2*phiM*phiM*(xe*g-c*c)/(ionConst*ionConst)+2*v2*phiM*(xe*g -c*c)/ionConst + xe/ionConst +v2*g)
        den = v2*phiM*phiM*(xe*g-c*c)/ionConst + phiM*(xe/ionConst + v2*g) + 1

        return num/den
    upper,lower = 0,np.inf
    result = integrate.quad(dfpintegrand, lower, upper, args=(Y,phiM,phiS,), limit=iterlim)
    dfp_dphi = result[0] / (4 * np.pi * np.pi)

    return ds_dphi+dfion_dphi+dfp_dphi

#####################SECOND DERIVATIVE FREE ENERGIES 2 VERSIONS#############################
def d2_FGauss_Y(Y,phiM,phiS,x = None, dx = None, ddx = None):
    x = x_solver(phiM, Y) if x == None else x
    dx = d1_x_solver(phiM, Y) if dx == None else dx
    ddx = d2_x_solver(phiM, Y) if ddx == None else ddx

    d2s = 0
    #################Entropyd2##########
    if phiM != 0:
        if phiM != (phiS - 1) / (-1 * qc - 1):
            d2s = 1 / (N * phiM) + qc / phiM + ((-qc - 1) ** 2) / (-1 * qc * phiM - phiS - phiM + 1)
        elif phiM == (phiS - 1) / (-1 * qc - 1):
            d2s = qc / phiM + 1 / (N * phiM)
    else:
        d2s = (-1 * qc - 1) ** 2 / (-1 * phiS + 1)

    d2fion_pt1num = qc * qc * Y * (np.sqrt(Y / (qc * phiM + phiS)) + 4 * np.sqrt(np.pi))
    d2fion_pt1den = 8 * np.sqrt(np.pi) * (qc * phiM + phiS) * (
            2 * np.sqrt(np.pi) * np.sqrt(Y * (qc * phiM + phiS)) + Y) ** 2
    d2fion_pt2 = (-1 * qc * qc) / (8 * np.sqrt(np.pi) * np.sqrt(Y) * (qc * phiM + phiS) ** (3 / 2))
    d2fion = d2fion_pt1num / d2fion_pt1den + d2fion_pt2

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
        ionConst = k * k * Y / (4 * np.pi) + phiS + qc * phiM
        v2 = (4 * np.pi / 3) * np.exp(-1* k * k / 6)
        D2BIG = g * d2xe * dx * dx + g * d1xe * ddx + 2 * d1xe * d1g * dx * dx + xe * d2g * dx * dx + xe * d1g * ddx - 2 * c * d2c * dx * dx - 2 * c * d1c * ddx - 2 * d1c * d1c * dx * dx
        secondOrder = xe * g - c * c

        Num1 = (-2 * qc * v2 * phiM * phiM * (g * d1xe * dx + xe * d1g * dx - 2 * c * d1c * dx) / (
                    ionConst * ionConst) + 4 * v2 * phiM * (
                            g * d1xe * dx + xe * d1g * dx - 2 * c * d1c * dx) / ionConst + 2 * d1xe * dx / ionConst + phiM * (
                            d2xe * dx * dx / ionConst + d1xe * ddx / ionConst - 2 * qc * d1xe * dx / (
                                ionConst * ionConst) + 2 * qc * qc * xe / (ionConst * ionConst * ionConst) + v2 * (
                                        d2g * dx * dx + d1g * ddx)) + v2 * phiM * phiM * (
                    D2BIG) / ionConst + 2 * qc * qc * v2 * phiM * phiM * (secondOrder) / (
                            ionConst * ionConst * ionConst) - 4 * qc * v2 * phiM * (secondOrder) / (
                            ionConst * ionConst) + 2 * v2 * secondOrder / ionConst - 2 * qc * xe / (
                            ionConst * ionConst) + 2 * v2 * d1g * dx)
        Den = (v2 * phiM * phiM * secondOrder / ionConst + phiM * (xe / ionConst + v2 * g) + 1)
        Num2 = (v2 * phiM * phiM * (g * d1xe * dx + xe * d1g * dx - 2 * c * d1c * dx) / ionConst + phiM * (
                    d1xe * dx / ionConst - qc * xe / (
                        ionConst * ionConst) + v2 * d1g * dx) - qc * v2 * phiM * phiM * secondOrder / (
                            ionConst * ionConst) + 2 * v2 * phiM * secondOrder / ionConst + xe / ionConst + v2 * g)
        return k*k* (Num1 / Den - (Num2 * Num2) / (Den * Den))

    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(d2_FP_toint, lowerlim, upperlim, args=(Y, phiM,), limit=iterlim)

    d2fp = result[0] / (4 * np.pi * np.pi)
    print(d2s,d2fp,d2fion, Y)
    return (d2s + d2fp + d2fion)
def d2_Fgauss_phiM(phiM,Y,phiS,x=None,dx=None,ddx=None):
    x = x_solver(phiM,Y) if x ==None else x
    dx = d1_x_solver(phiM,Y) if dx ==None else dx
    ddx = d2_x_solver(phiM,Y) if ddx ==None else ddx

    d2s =0
    #################Entropyd2##########
    if phiM!=0:
        if phiM != (phiS-1)/(-1*qc -1):
            d2s = 1/(N*phiM)+ qc/phiM + ((-qc-1)**2)/(-1*qc*phiM-phiS-phiM + 1)
        elif phiM == (phiS-1)/(-1*qc -1):
            d2s = qc/phiM + 1/(N*phiM)
    else: d2s = (-1*qc -1)**2/(-1*phiS +1)

    d2fion_pt1num = qc * qc * Y * (np.sqrt(Y / (qc * phiM + phiS)) + 4 * np.sqrt(np.pi))
    d2fion_pt1den = 8 * np.sqrt(np.pi) * (qc * phiM + phiS) * (
                2 * np.sqrt(np.pi) * np.sqrt(Y * (qc * phiM + phiS)) + Y) ** 2
    d2fion_pt2 = (-1 * qc * qc) / (8 * np.sqrt(np.pi) * np.sqrt(Y) * (qc * phiM + phiS) ** (3 / 2))
    d2fion = d2fion_pt1num / d2fion_pt1den + d2fion_pt2
    #################Electrofreeenergyd2###########

    def d2_FP_toint(k, Y, phiM):
        xe = xee(k, x, sigS=sigShift_xe)
        d1xe = d1_xee(k, x, sigS=sigShift_xe)
        d2xe = d2_xee(k,x,sigS=sigShift_xe)
        g = gk(k, x)
        d1g = d1_gk(k, x)
        d2g = d2_gk(k,x)
        c = ck(k, x, sigs=sigShift_ck)
        d1c = d1_ck(k, x, sigs=sigShift_ck)
        d2c = d2_ck(k,x,sigs=sigShift_ck)
        ionConst = k * k * Y / (4 * np.pi) + phiS + qc * phiM
        v2 = (4 * np.pi / 3) * np.exp(-k * k / 6)
        D2BIG = g*d2xe*dx*dx + g*d1xe*ddx + 2*d1xe*d1g*dx*dx + xe*d2g*dx*dx + xe*d1g*ddx - 2*c*d2c*dx*dx - 2*c*d1c*ddx - 2*d1c*d1c*dx*dx
        secondOrder = xe*g-c*c
        Num1 = (-2*qc*v2*phiM*phiM*(g*d1xe*dx+ xe*d1g*dx - 2*c*d1c*dx)/(ionConst*ionConst) + 4*v2*phiM*(g*d1xe*dx + xe*d1g*dx - 2*c*d1c*dx)/ionConst + 2*d1xe*dx/ionConst + phiM*(d2xe*dx*dx/ionConst + d1xe*ddx/ionConst -2*qc*d1xe*dx/(ionConst*ionConst)+2*qc*qc*xe/(ionConst*ionConst*ionConst)+v2*(d2g*dx*dx+d1g*ddx))+ v2*phiM*phiM*(D2BIG)/ionConst + 2*qc*qc*v2*phiM*phiM*(secondOrder)/(ionConst*ionConst*ionConst) - 4*qc*v2*phiM*(secondOrder)/(ionConst*ionConst) + 2*v2*secondOrder/ionConst - 2*qc*xe/(ionConst*ionConst)+ 2*v2*d1g*dx)
        Den = (v2*phiM*phiM*secondOrder/ionConst + phiM*(xe/ionConst+ v2*g) +1)
        Num2 = (v2*phiM*phiM*(g*d1xe*dx + xe*d1g*dx - 2*c*d1c*dx)/ionConst + phiM*(d1xe*dx/ionConst - qc*xe/(ionConst*ionConst) +v2*d1g*dx) - qc*v2*phiM*phiM*secondOrder/(ionConst*ionConst) + 2*v2*phiM*secondOrder/ionConst + xe/ionConst + v2*g)
        return (Num1/Den - Num2*Num2/(Den*Den))*k*k
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(d2_FP_toint, lowerlim, upperlim, args=(Y, phiM,), limit=iterlim)
    d2fp= result[0] / (4 * np.pi * np.pi)
    return np.sqrt((d2s + d2fp + d2fion) ** 2)


def getSpinodal(phiMs):
    Ys=[]
    for j in range(len(phiMs)):
        i = phiMs[j]
        y = root_scalar(d2_FGauss_Y, args=(i, phiS,), x0=.5, bracket=[1e-3, .99])
        print(i,y.root)
        Ys.append(y.root)
    return Ys
def spin_yfromphi(phiM,phiS,guess):
    guess1 = guess
    #y = fsolve(d2_FGauss_Y, args=(phiM, phiS,), x0=guess1)
    y = brenth(d2_FGauss_Y,.01, 10, args=(phiM,phiS,))
    print('phi', phiM, 'l/lb', y)
    return -1*y
def findCrit(phiS,guess):

    bounds = [(.005,.05)]

    ###FROM LIN
    phi_max = (1 - 2 * phiS) / (1 + qc)
    ini1, ini3, ini2 = 1e-6, 1e-2, phi_max*2/3
    Yc = minimize(spin_yfromphi, x0=guess, args=(phiS, guess,), method='Nelder-Mead', bounds=bounds)
    #result = brent(spin_yfromphi, args=(phiS,guess), brack = (ini1,ini3,ini2), full_output=1)
   # phiC,Yc = result[0], result[1]
    phiC = Yc.x
    #phiC = Yc
    return phiC, -1*Yc.fun
    #return phiC, Yc
phiC,Yc = findCrit(phiS, guess=.02)
print(phiC,Yc)

def findSpinlow(Y,phiC):
    initial = phiC/2
    bounds = [(epsilon, phiC-epsilon)]
    #result = minimize(FreeEnergyD2reverse, initial, args=(Y,phiS,),method='Powell',bounds=bounds)
    result = minimize(d2_Fgauss_phiM, initial, args=(Y, phiS,), method='Powell', bounds=bounds)
    #result = fsolve(FreeEnergyD2reverse, x0=initial, args=(Y,phiS))
    return result.x
def findSpinhigh(Y,phiC):
    initial = phiC+ phiC/2
    bounds = [(phiC+epsilon, 1-epsilon)]
    #result = minimize(FreeEnergyD2reverse, initial, args=(Y,phiS),method='Powell',bounds=bounds)
    result = minimize(d2_Fgauss_phiM, initial, args=(Y, phiS), method='Nelder-Mead', bounds=bounds)

    #result = fsolve(FreeEnergyD2reverse, x0=initial, args=(Y,phiS))
    return result.x

def FBINODAL(variables,Y,phiBulk):
    phi1,phi2 = variables
    v = (phi2-phiBulk)/(phi2-phi1)


    #eqn = v * ftot(phi1,Y,phiS) + (1-v)*ftot(phi2,Y,phiS)
    eqn = v * ftot_gaussIons(phi1, Y, phiS) + (1 - v) * ftot_gaussIons(phi2, Y, phiS)
    #return eqn
    return T0*(eqn - ftot_gaussIons(phiBulk,Y,phiS))
def getInitialVsolved(Y,spinlow,spinhigh,phiBulk):
    bounds = [(epsilon,spinlow-epsilon),(spinhigh+epsilon, 1-epsilon)]
    initial_guess=(spinlow*.899, spinhigh*1.101)
    result = minimize(FBINODAL, initial_guess, args=(Y, phiBulk), method='Nelder-Mead', bounds=bounds)
    #result = minimize(totalFreeEnergyVsolved, initial_guess,args=(Y,),method='Powell',bounds=bounds)
    phi1i,phi2i= result.x
    return phi1i,phi2i
def makeconstSLS(Y,phiBulk):
    def seperated(variables):
        return FBINODAL(variables, Y, phiBulk) - ftot_gaussIons(phiC, Y, phiS)
    return [{'type': 'ineq', 'fun': seperated}]
def minFtotal(Y,phiC,lastphi1,lastphi2):

    phi1spin = findSpinlow(Y, phiC)[0]
    phi2spin = findSpinhigh(Y, phiC)[0]
    print(lastphi1, lastphi2, 'last 1&2')
    print(phi1spin, phi2spin, 'SPINS LEFT/RIGHT')
    phiB = (phi1spin+phi2spin)/2

    phi1i, phi2i = getInitialVsolved(Y, phi1spin, phi2spin,phiB)
    initial_guess=(phi1i,phi2i)
    #initial_guess=(phi1spin*.9,phi2spin*1.1)
    #initial_guess=(phi1spin*.899,phi2spin*1.301)
    #phi2Max = (1-2*phiS)/(1+qc)#########FROM LIN CODE GITHUB ????


    bounds = [(epsilon, phi1spin - epsilon), (phi2spin+epsilon, 1-epsilon)]
    maxL = minimize(FBINODAL, initial_guess, args=(Y, phiB), method='L-BFGS-B', jac=Jac_fgRPA, bounds=bounds, options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20})
    #maxL = minimize(totalFreeEnergyVsolved,initial_guess,args=(Y,phiB),method='SLSQP',jac=Jac_fgRPA,bounds=bounds,options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20} )
    #maxL = minimize(totalFreeEnergyVsolved,initial_guess,args=(Y,phiB),method='Newton-CG',jac=Jac_fgRPA,bounds=bounds,options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20} )

    #const = makeconstSLS(Y)
    #maxL = minimize(totalFreeEnergyVsolved, initial_guess, args=(Y,), method='SLSQP', constraints=const,bounds=bounds)
    #maxL = minimize(totalFreeEnergyVsolved, initial_guess, args=(Y,), method='Powell', bounds=bounds)
    #maxparams = maxL.x
    #phi1min,phi2min = maxparams
    phi1min = min(maxL.x)
    phi2min = max(maxL.x)
    #v = (phiC - phi2min)/(phi1min-phi2min)
    v = (phi2min-phiB)/(phi2min-phi1min)

    print('\nwe have finished minimizing for Y = ',Y, 'just cuz curious: phi1,phi2, v = ',phi1min,phi2min,v)

    return phi1min,phi2min
def Jac_fgRPA(vars,Y,phiB):
    phi1=vars[0]
    phi2=vars[1]
    v = (phi2-phiB)/(phi2-phi1)

    f1 = ftot_gaussIons(phi1,Y,phiS)
    f2 = ftot_gaussIons(phi2,Y,phiS)
    df1 = d1ftotGauss_dphi(phi1,Y,phiS)
    df2 = d1ftotGauss_dphi(phi2,Y,phiS)

    J = np.empty(2)
    J[0] = v*( (f1-f2)/(phi2-phi1) + df1 )
    J[1] = (1-v)*( (f1-f2)/(phi2-phi1) + df2 )

    return J*T0
def getBinodal(Yc,phiC,minY):
    phibin=phiC
    Ybin = np.array([Yc])
    Ytest= Yc - scale_init

    Y_range= Yc - minY

    while Ytest>minY:
        Y_ratio_done = (Yc -Ytest)/Y_range
        #print(Ytest, "until", minY)
        phiLlast,phiDlast = phibin[0], phibin[-1]
        phi1,phi2 = minFtotal(Ytest, phiC, phiLlast, phiDlast)

        if phi1<phiLlast and phi2>phiDlast:
            phi1=np.array([phi1])
            phi2=np.array([phi2])
            phibin = np.concatenate((phi1, phibin, phi2))
            Ybin = np.concatenate(([Ytest], Ybin, [Ytest]))
        else: print('someglitch, repeating with a skipped step')
        ####HIGHER RESOLUTION AT TOP OF PHASE DIAGRAM###################
        #resolution = scale_init * np.exp((Yc / Ytest) ** 3) / np.exp(1)
        resolution = scale_init *(1 + Y_ratio_done*(scale_final/scale_init))
        print("NEXT YTEST CHANGED BY:", resolution, "and Ytest=", Ytest)
        Ytest-=resolution

    return phibin,Ybin
