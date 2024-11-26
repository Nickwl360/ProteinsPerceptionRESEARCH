
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
from Utilities.EQUILIBRIUMFINDER import EquilibriumMatrix

import os


current_dir = os.path.dirname(os.path.abspath(__file__))
example_file_path = os.path.join(current_dir, 'mc_perception_opencl.cl')
perception_cl_prog = example_file_path

# constants
Tmax = 10_000
MAXTOP=5
MAXBOT = 26
rng = np.random



def calc_next_state(params,current_state, prog_path):

    #halpha, ha, hbeta,hb,hgamma,hc,hdelta,hd, kcoop,kcomp,kdu,kud, kx = params
    # device = cl.get_platforms()[0].get_devices()[0]
    # ctx = cl.Context([device])
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    Result = np.zeros((MAXTOP * MAXTOP * MAXBOT * MAXBOT), dtype=np.float64)
    Result_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, Result.nbytes)

    params = np.array(params, dtype=np.float64)
    params_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=params)

    current_state = np.array(current_state, dtype=np.float64)
    current_state_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_state)


    with open(prog_path, "r") as f:
        program_source = f.read()
    program = cl.Program(ctx, program_source).build()

    global_size = ((MAXTOP*MAXTOP*MAXBOT*MAXBOT),)
    calc = program.calc_next_state

    try:
        calc(queue, global_size, None, Result_buf, params_buf, current_state_buf)
        queue.finish()
    except cl.LogicError as e:
        print("Error during kernel execution:", e)
        return None

    # Read the results back from the GPU to the host
    try:
        cl.enqueue_copy(queue, Result, Result_buf)
    except cl.LogicError as e:
        print("Error copying buffer to host:", e)
        return None
    return Result

def get_Pij(params):
    Pij = np.zeros((MAXTOP,MAXTOP,MAXBOT,MAXBOT,MAXTOP,MAXTOP,MAXBOT,MAXBOT),dtype=np.float64)
    for A in range(0,MAXTOP):
        for B in range(0,MAXTOP):
            print(A,B)
            for C in range(0,MAXBOT):
                for D in range(0,MAXBOT):
                    state = (A,B,C,D)
                    P_next = calc_next_state(params, state, perception_cl_prog)
                    P_next/=np.sum(P_next)
                    P_next= P_next.reshape((MAXTOP,MAXTOP,MAXBOT,MAXBOT))
                    Pij[A,B,C,D,:,:,:,:] = P_next

    return Pij


def renormalize(Pijkl):
    # Create a matrix to store the normalization factors
    normalizefactors = np.sum(Pijkl, axis=(4, 5, 6, 7))  # Sum over the end state indices
    # Ensure that the denominator is not zero to avoid division by zero
    normalizefactors[normalizefactors == 0] = 1.0
    # Normalize the transition matrix
    Pijkl /= normalizefactors[:, :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    #This works, I double checked with the old method
    return Pijkl


def nextNs(P,na,nb,nc,nd):
    # print(na,nb,nc,nd)
    Parr = P[na,nb,nc,nd,:,:,:,:]
    randnum = rng.random()
    SUM = 0
    NAm = 0
    NBn = 0
    NCo = 0
    NDp = 0
    for i in range(MAXTOP):
        for j in range(MAXTOP):
            for k in range(MAXBOT):
                for l in range(MAXBOT):
                    SUM += Parr[i][j][k][l]
                    if SUM >= randnum:
                        NAm = i
                        NBn = j
                        NCo = k
                        NDp = l
                        #print(type(NAm), type(NBn), type(NCo), type(NDp),NAm,NBn,NCo,NDp, 'types A-D')
                        return NAm,NBn,NCo,NDp
    if type(NAm)!= type(None) and type(NBn)!= type(None) and type(NCo)!= type(None) and type(NDp)!= type(None):
        return NAm,NBn,NCo,NDp
    else: return 0,0,0,0

def faster_function(Parr,ns):
    na,nb,nc,nd=ns
    Parr = Parr[na, nb, nc, nd, :, :, :, :]
    randnum = rng.random()

    shape = Parr.shape
    flat_Parr = Parr.reshape(-1)  # Flatten the Parr array
    cumsum = np.cumsum(flat_Parr)  # Compute cumulative sum
    index = np.searchsorted(cumsum, randnum)  # Find index where randnum fits in cumsum

    if index < len(cumsum):
        NAm, NBn, NCo, NDp = np.unravel_index(index, shape)
        return NAm, NBn, NCo, NDp
    else:
        return 0, 0, 0, 0
def simulation(Nstart,pmnopnorm,Tmax):
    NA = Nstart[0]
    NB = Nstart[1]
    NC = Nstart[2]
    ND = Nstart[3]
    t = 0
    A = [NA]
    B = [NB]
    C = [NC]
    D = [ND]

    pmnopnormal = pmnopnorm
    while t < Tmax:
        #print(params)
        #NA,NB,NC,ND = nextNs(pmnopnormal,NA,NB,NC,ND)
        NA,NB,NC,ND=faster_function(pmnopnormal,(NA,NB,NC,ND))
        t += 1
        print(t)
        # print(NA, NB, NC, ND, 'a,b,c,d')

        A.append(NA)
        B.append(NB)
        C.append(NC)
        D.append(ND)
        # print("Last A:", A[len(A) - 1])
        # print("Last B:", B[len(B) - 1])
        # print("Last C", C[len(C) - 1])
        # print("Last D:", D[len(D) - 1])
        # print(type(A), type(B), type(C),type(D), 'types A-D')
    return A,B,C,D



########PARAM EDITING#############################
initial=(0,0,0,0)  #A,B,C,D
######OLD VERSION FOR REFERENCE
#ULC = 0.8245
#LLC = 0.297
#params = ( -1 * ULC + epsilon1/2,ULC + -1*epsilon1/2, -1*LLC +epsilon2/2,LLC + -1*epsilon2/2, kcoop, kcomp,kdu,kud,kx)
#(halpha, ha,hgamma,hc, kcoop, kcomp,kdu,kud,kx) = ( -1 * ULC + epsilon1/2,ULC + -1*epsilon1/2, -1*LLC +epsilon2/2,LLC + -1*epsilon2/2,2.0,2.43,.8175,.1681,.4359)
#(halpha, ha,hgamma,hc, kcoop, kcomp,kdu,kud,kx) = ( -1 * ULC + epsilon1/2,ULC + -1*epsilon1/2, -1*LLC +epsilon2/2,LLC + -1*epsilon2/2,kcoop,kcomp,kdu,kud,kx)

epsilon1= .0
epsilon2 = .0
# kcoop,kcomp,kdu,kud,kx

#CalcedHsStrong= (-5.836,-6.09057,-3.8672,6.9068)
#DT = .01stuff
#set15, 9inference,dt=.01 seeded with calculations  L = 0.3089
#(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) =(-6.00432578 ,-6.03486149, -3.80626702 , 5.35061176 , 6.4340547  , 3.67099913 ,8.6066445  , 0.28647123 , 2.42905138)
#set0, 9inference, dt=.01 L = .1915
#(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx)=(-6.58151137, -5.4149016 , -7.32583317, 11.50703375,  8.02477748,  7.15952577, 11.87505467,  0.31408652,  5.4198252 )

#I=1 dt001 L = .05868?
#(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx)=(-8.28640719 ,-8.42166917 ,-6.27641919 ,-0.61254556  ,4.71567638  ,2.15322098,6.15779811  ,0.29307981  ,1.63738961)

#I=.0625 dt001 L = .0403
#(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx)= (-8.96733557, -7.73231853, -6.01935508 ,-0.99322105,  4.7228139 ,  1.98114397 ,6.05944224 , 0.29747507 , 1.53067954)

#I = .375 dt001 L = 0.04606900201790876
#(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx)=(-8.57134921, -8.1347204,  -6.11845674 ,-0.84250452,  4.60511341,  2.04416943,  5.97552154,  0.29414426 , 1.57459083)

#I = .6875 dt001  L = 0.05103779273020195
(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx)=(-8.39780022, -8.31815575 ,-6.24283186, -0.62797361,  4.64786633,  2.1348466,  6.06874194 , 0.29438665 , 1.62159095)

params = (halpha, ha, halpha - epsilon1, ha + epsilon1,hgamma,hc,hgamma-epsilon2,hc +epsilon2, kcoop, kcomp,kdu,kud,kx)
#
if __name__ == "__main__":
    Pij = get_Pij(params)

    ##RUNNINGFORWARD################################################
    As,Bs,Cs,Ds = simulation(initial,Pij,Tmax)

    #############SAVING###################################
    total_length = len(As)

    print('halpha, hA,hbeta,hB,hgamma,hC,hdelta,hD,kcoop,kcomp,kdu,kud,kx = ',params)

    ts= np.linspace(0,Tmax,len(As))
    plt.plot(ts,As,linewidth=1,c='b')
    plt.plot(ts,Bs,linewidth=1,c='r')
    plt.xlabel("T",fontsize=15)
    plt.ylabel("# Activated",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-.1,4.5)
    plt.show()
    # # #
    # plt.figure()
    # plt.plot(ts, Cs, linewidth=1, c='b')
    # plt.plot(ts, Ds, linewidth=1, c='r')
    # plt.xlabel("T" , fontsize=15)
    # plt.ylabel("# Activated", fontsize=15)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.show()
    # # #
    # np.save('JochenI_1dt_.001HseedA',As)
    # np.save('JochenI_1dt_.001HseedB',Bs)
    # np.save('JochenI_1dt_.001HseedC',Cs)
    # np.save('JochenI_1dt_.001HseedD',Ds)



