
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import global_params as gp
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
example_file_path = os.path.join(current_dir, 'mc_perception_opencl.cl')
perception_cl_prog = example_file_path

# constants
Tmax = 10_000
MAXTOP=gp.MAXTOP
MAXBOT = gp.MAXBOT
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




#
if __name__ == "__main__":
    ########PARAM EDITING#############################
    ### TO REPLACE WITH A SAVING FUNCTION ###
    initial = (0, 0, 0, 0)  # A,B,C,D

    epsilon1 = .0
    epsilon2 = .0

    (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = (-8.39780022, -8.31815575, -6.24283186, -0.62797361, 4.64786633, 2.1348466, 6.06874194, 0.29438665, 1.62159095)

    params = (halpha, ha, halpha - epsilon1, ha + epsilon1, hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)
    Pij = get_Pij(params)

    ########################################################

    p_test = calc_next_state((params),initial, perception_cl_prog)
    print(p_test)
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



