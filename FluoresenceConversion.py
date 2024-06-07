from OldMCProjects.MaxCalAAsim import *
from numba import jit
from scipy.optimize import minimize



MAX=75
params = (-.512,.585,.0298)
m =6

###CORRUPTER####
# data = np.load('NA-AAtest100k.npy')
F_f0 = 100 # mean fluorenscence readout for a single protein
F_sig = 30
# ITS_pull = lambda N, p: N*F_f0 + F_sig*np.sqrt(2*N)*erfinv(2*p-1)
# prob = np.random.rand()
# fdata = ITS_pull(data, prob)
# np.save('AAflur100k',fdata)

mcdata = np.load('OldMCProjects/NA-AAtest100k.npy')
dflur =np.load('AAflur100k.npy')

@jit(nopython=True)
def Pfn(f,a,b,n):
    if n>0:
        pfn = 1. / (np.sqrt(2. * np.pi*b**2*n)) * np.exp(-np.power((f - n*a),2)/(2*n*b**2))
    else: pfn = 0
    return pfn

@jit(nopython=True)
def Pnequ(p):
    plarge = np.linalg.matrix_power(p,1000)
    i = np.zeros(MAX)
    i[0]=1
    pequ = i @ plarge
    return pequ
@jit(nopython=True)
def phi(pneq,f):
    phi = np.zeros(MAX)
    if f !=0:
        for i in range(MAX):
            phi[i]= Pfn(f,F_f0,F_sig,i) * pneq[i]
        phi = phi/sum(phi)
    else:
        phi[0]=1

    return phi

@jit(nopython=True)
def fluLike(params,flurdata,m):
    L=0
    ha,hA,ka = params
    pij = calcPij(ha,hA,ka)
    pijm = np.linalg.matrix_power(pij,m)
    peq = Pnequ(pij)
    for f in range(len(flurdata)-m):
        phi1 = phi(peq,flurdata[f])
        phi2 = phi(peq,flurdata[f+m])
        v = phi1 @ pijm @ phi2
        L+= np.log(v)
    print(-1*L)
    return -1* L

def maximize(data,m):
    initial = np.array([-0.511, 0.586, 0.0288])
    maxL = minimize(fluLike, initial, args=(data,m,), method='Nelder-Mead')
    maxparams = maxL.x
    return maxparams
print(maximize(dflur,m))