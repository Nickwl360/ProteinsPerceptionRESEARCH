import numpy as np
import pyopencl as cl
from itertools import product

MAXTOP=5
MAXBOT=12
BI=2

Pls_prog = '/Users/Nickl/PycharmProjects/Researchcode (1) (1)/MCPLs.cl'

def getPls(params, prog_path):
    halpha, ha, hbeta,hb,hgamma,hc,hdelta,hd, kcoop,kcomp,kdu,kud, kx = params
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    with open(prog_path, "r") as f:
        program_source = f.read()
    program = cl.Program(ctx, program_source).build()
    #NextPmnop = np.zeros(((MAXTOP*MAXTOP*MAXBOT*MAXBOT)**2),dtype=np.float64)
    Pls = np.zeros((MAXTOP*MAXTOP*MAXBOT*MAXBOT*BI*BI*BI*BI*BI*BI*MAXTOP*MAXTOP),dtype=np.float64)
    Pls_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, Pls.nbytes)

    global_size = ((MAXTOP*MAXTOP*MAXBOT*MAXBOT*BI*BI*BI*BI*BI*BI*MAXTOP*MAXTOP),)
    calc = program.compute_Pls
    calc.set_scalar_arg_dtypes([None, np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64])
    calc(queue, global_size, None, Pls_buf, halpha, hbeta, hgamma, hdelta,ha,hb,hc,hd, kcomp, kcoop,kdu,kud,kx)
    queue.finish()
    # Read the results back from the GPU to the host
    cl.enqueue_copy(queue, Pls, Pls_buf)
    return Pls

def renormalize(Plsreshaped):
    # Create a matrix to store the normalization factors
    normalizefactors = np.sum(Plsreshaped, axis=(4,5,6,7,8,9,10,11))  # Sum over the end state indices

    # Ensure that the denominator is not zero to avoid division by zero
    zero_indices = np.where(normalizefactors == 0)
    normalizefactors[zero_indices] = 1.0  # Set normalization factors to 1.0 for states with zero sum

    # Normalize the transition matrix
    Plsreshaped /= normalizefactors[:,:,:,:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis,np.newaxis]
    # Reshape the normalization factors to match the shape of Plsreshaped

    # Normalize the transition matrix using each set of normalization factors

    #This works, I double checked with the old method

    return Plsreshaped

def calcRateMC(Pnorm,NR):
    ##########[na,nb,nc,nd, ll+, ll-, lu+, ll+',ll-',lu+', lu-,lu-']
    num = np.sum(Pnorm[NR,:,:,:,1,:,:,:,:,:,:,:])
    den = np.sum(Pnorm[NR,:,:,:,0,:,:,:,:,:,:,:])
    print(num,den)
    dt = .001
    rate = (1/dt)*((num)/(num+den))
    return rate

def calcRateJoch(I,NR):
    wvis,wpred,thetae = 1.7798,2.34,-1.6529
    ve = (1/1.949)
    uep = wvis*I - wpred * NR + thetae
    ratep = (ve/2)*np.exp(uep/2)
    return ratep

#################PARAMETERS#######################
epsilon1= .0
epsilon2 = .0
#i = .6875, dt = 001
(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx)=(-8.39780022, -8.31815575 ,-6.24283186, -0.62797361,  4.64786633,  2.1348466,  6.06874194 , 0.29438665 , 1.62159095)


params = (halpha, ha, halpha - epsilon1, ha + epsilon1,hgamma,hc,hgamma-epsilon2,hc +epsilon2, kcoop, kcomp,kdu,kud,kx)
pls = getPls(params,Pls_prog)
plsreshape =pls.reshape((MAXTOP,MAXTOP,MAXBOT,MAXBOT,BI,BI,BI,BI,BI,BI,MAXTOP,MAXTOP))
plsnorm = renormalize(plsreshape)

print(calcRateJoch(.6875,4), 'jochs I = .6875, NA = 4, vl+ rate')
print(calcRateMC(plsreshape,4), 'MC I = .6875, NA = 4, vl+rate')
