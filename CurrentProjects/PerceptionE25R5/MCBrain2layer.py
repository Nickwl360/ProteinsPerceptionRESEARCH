
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
#0from Utilities.EQUILIBRIUMFINDER import EquilibriumMatrix

import os


current_dir = os.path.dirname(os.path.abspath(__file__))
example_file_path = os.path.join(current_dir, 'MCBrain2layer.cl')
Pmnop_prog = example_file_path

# constants
Tmax = 10_000
MAXTOP=5
MAXBOT = 12
rng = np.random

###############SAVING DATA FILES############
chunk_size = 1000000
directory = 'InferedTrajectoriesMCBRAIN'
if not os.path.exists(directory):
    os.makedirs(directory)

def runtime_program(params, prog_path):
    halpha, ha, hbeta,hb,hgamma,hc,hdelta,hd, kcoop,kcomp,kdu,kud, kx = params
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    with open(prog_path, "r") as f:
        program_source = f.read()
    print('read')
    program = cl.Program(ctx, program_source).build()
    print('built')
    NextPmnop = np.zeros(((MAXTOP*MAXTOP*MAXBOT*MAXBOT)**2),dtype=np.float64)
    Pmnop_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, NextPmnop.nbytes)

    global_size = ((MAXTOP*MAXTOP*MAXBOT*MAXBOT)**2,)
    calc = program.compute_Pmnop
    calc.set_scalar_arg_dtypes([None, np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,np.float64])
    calc(queue, global_size, None, Pmnop_buf, halpha, hbeta, hgamma, hdelta,ha,hb,hc,hd, kcomp, kcoop,kdu,kud,kx)
    queue.finish()
    # Read the results back from the GPU to the host
    cl.enqueue_copy(queue, NextPmnop, Pmnop_buf)
    return NextPmnop
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
def getequilib(Pnorm):
    na, nb, nc, nd = (0,0,0,0)
    suma,sumb,sumc,sumd = (0,0,0,0)
    # initial = np.zeros((MAXTOP**2*MAXBOT**2))
    # initial[0]=1
    # Pfix = np.reshape(Pnorm,(MAXTOP**2*MAXBOT**2,MAXTOP**2*MAXBOT**2))
    # plarge = np.linalg.matrix_power(Pfix,100000)
    # pmnoequ = np.matmul(initial,plarge)
    pmnoequ = EquilibriumMatrix(Pnorm,(MAXTOP*MAXTOP*MAXBOT*MAXBOT),((MAXTOP*MAXTOP*MAXBOT*MAXBOT),(MAXTOP*MAXTOP*MAXBOT*MAXBOT)),(MAXTOP,MAXTOP,MAXBOT,MAXBOT))
    pequshape = pmnoequ.transpose()

    for i in range(MAXTOP):
        suma += np.sum(pequshape[i, :, :, :])
        sumb += np.sum(pequshape[:, i, :, :])
        na += i*np.sum(pequshape[i, :, :, :])
        nb += i*np.sum(pequshape[:, i, :, :])
    for i in range(MAXBOT):
        sumc += np.sum(pequshape[:, :, i, :])
        sumd += np.sum(pequshape[:, :, :, i])
        nc += i*np.sum(pequshape[:, :, i, :])
        nd += i*np.sum(pequshape[:, :, :, i])
    na /=suma
    nb /=sumb
    nc /=sumc
    nd /=sumd

    return na,nb,nc,nd
def calcaminusb(hgamma,hdelta):
    hg = hgamma
    hd = hdelta
    e1 = .0
    e2 = hg - hd
    # (halpha, ha, hgamma, hc, kcoop, kcomp, kdu, kud, kx) = (e1 / 2, -1 * e1 / 2, e2 / 2, -1 * e2 / 2, 2.35,2.65,.55,1.75,.75)
    # paramsp = (halpha, ha, halpha - e1, ha + e1, hg, hc, hd, hc + e2, kcoop, kcomp, kdu, kud,kx)
    initial = (0, 4, 0, 0)  # A,B,C,D
    ULC = 0.8245
    LLC = 0.297
    epsilon1 = .0
    epsilon2 = 0.0  # kcoop,kcomp,kdu,kud,kx
    (halpha, ha, hgamma, hc, kcoop, kcomp, kdu, kud, kx) = (-1 * ULC + epsilon1 / 2, ULC + -1 * epsilon1 / 2, -1 * LLC + epsilon2 / 2, LLC + -1 * epsilon2 / 2, 2.0, 2.43, .8175, .1681, .4359)
    paramsp = (halpha, ha, halpha - epsilon1, ha + epsilon1, hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)

    Pmnop = runtime_program(paramsp, Pmnop_prog)
    pmnopreshape = Pmnop.reshape((MAXTOP, MAXTOP, MAXBOT, MAXBOT, MAXTOP, MAXTOP, MAXBOT, MAXBOT))
    pmnopnormal = renormalize(pmnopreshape)
    a,b,c,d = getequilib(pmnopnormal)
    print(hg,hd)
    return (a-b)/(MAXTOP-1)
def movingAvg(dataset,window):
    movingavg = []
    for i in range(len(dataset)):
        sum = 0
        num = 0
        sum+= dataset[i]
        num+=1
        for n in range(1,window+1):
            if i - n >= 0:
                sum+= dataset[i-n]
                num+=1
        maxwindow = num -1
        for n in range(1,maxwindow+1):
            if i +n < len(dataset) and maxwindow!= 0:
                sum+= dataset[i+n]
                num+=1
        sum/=num
        movingavg.append(sum)
    return movingavg
def lowerTransitions(NA, probs):

    ###CALC PROBABILITIES HERE
    lowerTrans= np.zeros((MAXBOT,MAXBOT))
    for i in range(MAXBOT):
        print(i)
        for j in range(MAXBOT):
            lowerTrans[i,j]=np.sum(probs[:,:,j,:,NA,:,i,:])

    norm = lowerTrans/np.sum(lowerTrans)
    return norm

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
    Pmnop = runtime_program(params,Pmnop_prog)
    pmnopreshape = Pmnop.reshape((MAXTOP, MAXTOP, MAXBOT, MAXBOT, MAXTOP, MAXTOP, MAXBOT, MAXBOT))
    pmnopnormal = renormalize(pmnopreshape)

    # NA4Stronglowertrans = lowerTransitions(4, pmnopnormal)
    # plt.imshow(NA4Stronglowertrans, cmap='viridis', interpolation='nearest')
    # plt.xlabel('NCt')
    # plt.ylabel('NCt+dt')
    # plt.show()

    ##GETTINGEQUILIB###################################################
    #print('<NA>,<NB>,<NC>,<ND> = ',getequilib(pmnopnormal))

    ##RUNNINGFORWARD################################################
    As,Bs,Cs,Ds = simulation(initial,pmnopnormal,Tmax)

    #############SAVING###################################
    total_length = len(As)

    print('halpha, hA,hbeta,hB,hgamma,hC,hdelta,hD,kcoop,kcomp,kdu,kud,kx = ',params)
    #########GRAPHING THINGS ###################
    num_chunks = total_length // chunk_size
    if total_length % chunk_size != 0:
        num_chunks += 1
    # for i in range(num_chunks):
    #     start_idx = i * chunk_size
    #     end_idx = min((i + 1) * chunk_size, total_length)
    #     np.save(os.path.join(directory, f'JochenI_6875dt_.001HseedA_chunk{i}'), As[start_idx:end_idx])
    #     np.save(os.path.join(directory, f'JochenI_6875dt_.001HseedB_chunk{i}'), Bs[start_idx:end_idx])
    #     np.save(os.path.join(directory, f'JochenI_6875dt_.001HseedC_chunk{i}'), Cs[start_idx:end_idx])
    #     np.save(os.path.join(directory, f'JochenI_6875dt_.001HseedD_chunk{i}'), Ds[start_idx:end_idx])

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

    #############EQUILIBRIUMSTUFF###############
    # pmnoequ = EquilibriumMatrix(pmnopnormal, (MAXTOP * MAXTOP * MAXBOT * MAXBOT),((MAXTOP * MAXTOP * MAXBOT * MAXBOT), (MAXTOP * MAXTOP * MAXBOT * MAXBOT)),(MAXTOP, MAXTOP, MAXBOT, MAXBOT))
    # pequshape = pmnoequ.transpose()
    # x_index = 10
    # y_index = 1
    #
    # # Get the dimensions of the matrix
    # rows, cols, _, _ = pequshape.shape
    # print(rows,cols)
    #
    # # Create x and y coordinates
    # x = np.arange(0, cols)
    # y = np.arange(0, rows)
    #
    # equdiagram = np.zeros((len(x), len(y)))
    # for i in range(12):#a
    #     for j in range(12):#b
    #         print(i, j)
    #         equdiagram[i][j] = sum(pequshape[i,j,:,:])/sum(pequshape)

    # Create a meshgrid from x and y
    # X, Y = np.meshgrid(x, y)
    # Z = pequshape[:, :, x_index, y_index]
    # plt.contourf(X, Y, c=Z, cmap='viridis',levels = 10)
    # plt.matshow(equdiagram,origin='lower',extent=(min(x), max(x), min(y), max(y)))
    # plt.title(r'Equilibrium Distribution: $\varepsilon = 0$')
    # plt.xlabel('NA')
    # plt.ylabel('NB')
    # plt.show()



