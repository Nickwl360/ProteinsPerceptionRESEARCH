import numpy as np
from OldProteinProjects.SCDcalc import *
from scipy import integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from fgRPA_init import *
from scipy.optimize import brute as gridsearch


########################FUNCTIONS################################

#############SEQUENCE SPECIFIC CHARGE/XEE###########################
def getSigShift(qs):
    sigS = []
    for i in range(len(qs)-1):
        sigi=0
        for j in range(len(qs)-1):
            if (j+i)<= len(qs)-1:
                sigi += qs[j]*qs[j+i]
        if i ==0:
            sigS.append(sigi)
        if i!=0:
            sigS.append(2*sigi)
    return sigS
sigShift = getSigShift(qs)
def xee(k,sigS):
    xeesum=0
    for i in range(0,N-1):
        xeesum += sigS[i]*np.exp((-1/6)*(k*k)*i)
    return xeesum/N
#####################SECOND DERIVATIVE FREE ENERGIES 2 VERSIONS#############################
def FreeEnergyD2FEL(Y,phiM,phiS):
    d2s =0
    #################Entropyd2##########
    if phiM!=0:
        if phiM != (phiS-1)/(-1*qc -1):
            d2s = 1/(N*phiM)+ qc/phiM + ((-qc-1)**2)/(-1*qc*phiM-phiS-phiM + 1)
        elif phiM == (phiS-1)/(-1*qc -1):
            d2s = qc/phiM + 1/(N*phiM)
    else: d2s = (-1*qc -1)**2/(-1*phiS +1)

    #################Electrofreeenergyd2###########
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(felD2integrand,lowerlim,upperlim,args=(Y,phiM), limit=150)
    d2fel= result[0]/(4*np.pi*np.pi)
    return d2s + d2fel
def felD2integrand(k,Y,phiM):
    x = xee(k,sigS=sigShift)
    return -1 * k * k * (4 * np.pi / Y)**2 * (qc + x)**2 / \
           (((4 * np.pi / Y) * (phiS + (qc + x) * phiM) + (k * k))**2)
def FreeEnergyD2FP(Y,phiM,phiS):
    d2s =0
    #################Entropyd2##########
    if phiM!=0:
        if phiM != (phiS-1)/(-1*qc -1):
            d2s = 1/(N*phiM)+ qc/phiM + ((-qc-1)**2)/(-1*qc*phiM-phiS-phiM + 1)
        elif phiM == (phiS-1)/(-1*qc -1):
            d2s = qc/phiM + 1/(N*phiM)
    else: d2s = (-1*qc -1)**2/(-1*phiS +1)

    d2fion_pt1num=qc*qc*Y*(np.sqrt(Y/(qc*phiM+phiS))+4*np.sqrt(np.pi))
    d2fion_pt1den = 8*np.sqrt(np.pi)*(qc*phiM+phiS)*(2*np.sqrt(np.pi)*np.sqrt(Y*(qc*phiM+phiS))+Y)**2
    d2fion_pt2= (-1*qc*qc)/(8*np.sqrt(np.pi)*np.sqrt(Y)*(qc*phiM+phiS)**(3/2))
    #print(d2fion_pt1num, d2fion_pt1den, d2fion_pt2)
    d2fion = d2fion_pt1num/d2fion_pt1den + d2fion_pt2

    #################Electrofreeenergyd2###########
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(fpD2integrand,lowerlim,upperlim,args=(Y,phiM), limit=150)
    d2fp= result[0]/(4*np.pi*np.pi)
    #print('ent', d2s, 'fp', d2fp, 'fion', d2fion)
    return d2s + d2fp + d2fion
def fpD2integrand(k,Y,phiM):
    x = xee(k,sigS=sigShift)
    num = -16*np.pi*np.pi*x*k*k*(k*k*Y+ 4*np.pi*phiS)*(x*k*k*Y+4*np.pi*x*(2*qc*phiM+ phiS)+ 2*k*k*qc*Y+8*np.pi*qc*(qc*phiM+phiS))
    den = (k*k*Y+ 4*np.pi*(qc*phiM+ phiS))*(k*k*Y+ 4*np.pi*(qc*phiM+ phiS))*(4*np.pi*(phiM*(x+qc)+phiS)+k*k*Y)*(4*np.pi*(phiM*(x+qc)+phiS)+k*k*Y)
    return num/den
def FreeEnergyD2reverseFP(phiM,Y,phiS):
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
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(fpD2integrand,lowerlim,upperlim,args=(Y,phiM,),limit=150)
    d2fp= result[0] / (4 * np.pi * np.pi)
    return np.sqrt((d2s + d2fp + d2fion) ** 2)
def felintegrand(k,Y,phiM,phiS):
    x = xee(k,sigS=sigShift)
    return (k**2)*np.log(1 + ((4*np.pi)/(k**2*Y))*(phiS+(qc+x)*phiM))
def fel(phiM, Y, phiS):
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(felintegrand, lowerlim, upperlim, args=(Y, phiM,phiS), limit=150)
    fel = result[0] / (4 * np.pi * np.pi)
    return fel
def fpintegrand(k,Y,phiM,phiS):
    x = xee(k, sigS=sigShift)
    return k*k*np.log(1+ (phiM*x)/((k*k*Y/(4*np.pi))+phiS+qc*phiM))
def fp(phiM, Y, phiS):
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(fpintegrand, lowerlim, upperlim, args=(Y, phiM,phiS), limit=150)
    fel = result[0] / (4 * np.pi * np.pi)
    return fel
def fion(phiM,Y,phiS):
    kl = np.sqrt(4*np.pi*(phiS+qc*phiM)/Y)
    return (-1/(4*np.pi))*(np.log(1+kl)-kl + .5*kl*kl)
def entropy(phiM,phiS):
    phiC = qc*phiM
    phiW = 1-phiM-phiS-phiC
    #################FIGURE OUT LOGIC FOR 0s
    if phiS!= 0:
        return (phiM/N)*np.log(phiM) + phiS* np.log(phiS) + phiC*np.log(phiC) + phiW*np.log(phiW)
    else:return (phiM/N)*np.log(phiM)+ phiC*np.log(phiC) + phiW*np.log(phiW)
def ftot_pointIons(phiM,Y,phiS):
    return entropy(phiM, phiS) + fel(phiM, Y, phiS)
def ftot_gaussIons(phiM,Y,phiS):
    return entropy(phiM, phiS) + fion(phiM, Y, phiS) + fp(phiM, Y, phiS)
def dftotGauss_dphi(phiM,Y,phiS):
    ds_dphi =np.log(phiM)/N + 1/N - 1 + qc*np.log(qc*phiM) + (-1*qc-1)*np.log(1-qc*phiM -phiS - phiM)
    dfion_dphi= (-1*np.sqrt(np.pi)*qc*np.sqrt((qc*phiM+phiS)/Y))/(Y*(2*np.sqrt(np.pi)*np.sqrt((qc*phiM)/Y)+1))
    def dfpintegrand(k,Y,phiM,phiS):
        x = xee(k, sigS=sigShift)
        a = k*k*Y/(4*np.pi)
        return k*k * (x*(phiS+ a))/((qc*phiM+phiS+a)*((qc+x)*phiM+phiS+a))
    upper,lower = 0,np.inf
    result = integrate.quad(dfpintegrand,lower,upper,args=(Y,phiM,phiS,),limit=150)
    dfp_dphi = result[0] / (4 * np.pi * np.pi)

    return ds_dphi+dfion_dphi+dfp_dphi
def checkPotentials(phi1,phi2,Y,phiS):
    thresh = .09
    if(np.abs(dftotGauss_dphi(phi1,Y,phiS)-dftotGauss_dphi(phi2,Y,phiS)) < thresh):
        return True
    else: return False
def getSpinodal(phiMs):
    Ys=[]
    for j in range(len(phiMs)):
        i = phiMs[j]
        y = root_scalar(FreeEnergyD2FP, args=(i, phiS,), x0=.5, bracket=[1e-3, .99])
        print(i,y.root)
        Ys.append(y.root)
    return Ys
def SpinodalY(phiM,phiS,guess):
    guess1 = guess
    y = fsolve(FreeEnergyD2FP, args=(phiM, phiS,), x0=guess1)
    print('phi', phiM, 'l/lb', y)
    return -1*y
def findCrits(phiS,guess):
    phiC=0
    Yc=0
    bounds = [(.001,.49)]
    Yc = minimize(SpinodalY, x0=guess, args=(phiS,guess,),method='Powell', bounds=bounds)
    phiC = Yc.x
    return phiC, -1*Yc.fun
phiC,Yc = findCrits(phiS,guess=.025)
def findSpinlow(Y,phiC):
    initial = phiC/2
    bounds = [(epsilon, phiC-epsilon)]
    #result = minimize(FreeEnergyD2reverse, initial, args=(Y,phiS,),method='Powell',bounds=bounds)
    result = minimize(FreeEnergyD2reverseFP, initial, args=(Y,phiS,),method='Powell',bounds=bounds)
    #result = fsolve(FreeEnergyD2reverse, x0=initial, args=(Y,phiS))
    return result.x
def findSpinhigh(Y,phiC):
    initial = phiC+ phiC/2
    bounds = [(phiC+epsilon, 1-epsilon)]
    #result = minimize(FreeEnergyD2reverse, initial, args=(Y,phiS),method='Powell',bounds=bounds)
    result = minimize(FreeEnergyD2reverseFP, initial, args=(Y,phiS),method='Nelder-Mead',bounds=bounds)

    #result = fsolve(FreeEnergyD2reverse, x0=initial, args=(Y,phiS))
    return result.x
def FreeEnergyD2reverse(phiM,Y,phiS):
    d2s =0
    #################Entropyd2##########
    if phiM!=0:
        if phiM != (phiS-1)/(-1*qc -1):
            d2s = 1/(N*phiM)+ qc/phiM + ((-qc-1)**2)/(-1*qc*phiM-phiS-phiM + 1)
        elif phiM == (phiS-1)/(-1*qc -1):
            d2s = qc/phiM + 1/(N*phiM)
    else: d2s = (-1*qc -1)**2/(-1*phiS +1)

    #################Electrofreeenergyd2###########
    upperlim = np.inf
    lowerlim = 0
    result = integrate.quad(felD2integrand,lowerlim,upperlim,args=(Y,phiM,),limit=150)
    d2fel= result[0]/(4*np.pi*np.pi)
    return np.sqrt((d2s + d2fel)**2)
    #return d2s + d2fel
def totalFreeEnergyVsolved(variables,Y):
    phi1,phi2 = variables
    v = (phiC-phi2)/(phi1-phi2)
    #eqn = v * ftot(phi1,Y,phiS) + (1-v)*ftot(phi2,Y,phiS)
    eqn = v * ftot_gaussIons(phi1, Y, phiS) + (1 - v) * ftot_gaussIons(phi2, Y, phiS)
    return eqn

def getInitialVsolved(Y,spinlow,spinhigh):
    bounds = [(epsilon,spinlow-epsilon),(spinhigh+epsilon, 1-epsilon)]
    initial_guess=(spinlow*.9, spinhigh*1.15)
    result = minimize(totalFreeEnergyVsolved, initial_guess, args=(Y,), method='Nelder-Mead', bounds=bounds)
    #result = minimize(totalFreeEnergyVsolved, initial_guess,args=(Y,),method='Powell',bounds=bounds)
    phi1i,phi2i= result.x
    return phi1i,phi2i
def makeconstSLS(Y):
    def seperated(variables):
        return totalFreeEnergyVsolved(variables,Y) - ftot_gaussIons(phiC,Y,phiS)
    return [{'type': 'ineq', 'fun': seperated}]

def minFtotal(Y,phiC,lastphi1,lastphi2, last_2_phi1, last_2_phi2):
    phi1spin = findSpinlow(Y, phiC)[0]
    phi2spin = findSpinhigh(Y, phiC)[0]
    #dphi1 = abs(last_2_phi1 - lastphi1)
    #dphi2 = abs(lastphi2 - last_2_phi2)
    print(lastphi1, lastphi2, 'last 1&2')
    print(phi1spin, phi2spin, 'SPINS LEFT/RIGHT')
    phi1i, phi2i = getInitialVsolved(Y, phi1spin, phi2spin)
    initial_guess=(phi1i,phi2i)

    bounds = [(epsilon, phi1spin - epsilon), (phi2spin+epsilon, 1-epsilon)]
    const = makeconstSLS(Y)
    maxL = minimize(totalFreeEnergyVsolved, initial_guess, args=(Y,), method='SLSQP', constraints=const,bounds=bounds)

    #ranges= [(phi1spin/2, phi1spin*.9), (phi2spin*1.1, phi2spin*2)]
    #ranges= [(phi2spin/4, phi1spin-epsilon), (phi2spin+epsilon, phi2spin*2.5)]
    #print('THIS IS THE RANGES: ',ranges)
    #maxL = gridsearch(totalFreeEnergyVsolved, ranges=ranges, args=(Y,), Ns=25,full_output=True)

    #maxL = minimize(totalFreeEnergyVsolved, initial_guess, args=(Y,), method='Powell', bounds=bounds)
    maxparams = maxL.x
    phi1min,phi2min = maxparams
    # phi_min = maxL[0]
    # if isinstance(phi_min, np.ndarray) or isinstance(phi_min, (list, tuple)):
    #     phi1min, phi2min = phi_min
    # else:
    #     raise TypeError("The result is not iterable.")
    # #phi1min,phi2min = maxL[0]
    v = (phiC - phi2min)/(phi1min-phi2min)
    print('\nwe have finished minimizing for Y = ',Y, 'just cuz curious: phi1,phi2, v = ',phi1min,phi2min,v)

    return phi1min,phi2min

def getBinodal(Yc,phiC,minY):
    phibin=phiC
    Ybin = np.array([Yc])
    Ytest=Yc-scale
    while Ytest>minY:

        #print(Ytest, "until", minY)
        phiLlast,phiDlast = phibin[0], phibin[-1]
        phiL2last,phiD2last=phiC,phiC
        phi1,phi2 = minFtotal(Ytest, phiC, phiLlast, phiDlast, phiL2last,phiD2last)
        phi1=np.array([phi1])
        phi2=np.array([phi2])
        phibin = np.concatenate((phi1, phibin, phi2))
        Ybin = np.concatenate(([Ytest], Ybin, [Ytest]))
        ####HIGHER RESOLUTION AT TOP OF PHASE DIAGRAM###################
        resolution = scale*np.exp((Yc/Ytest))/np.exp(1)
        print("NEXT YTEST CHANGED BY:", resolution, "and Ytest=", Ytest)
        Ytest-=resolution

    return phibin,Ybin




