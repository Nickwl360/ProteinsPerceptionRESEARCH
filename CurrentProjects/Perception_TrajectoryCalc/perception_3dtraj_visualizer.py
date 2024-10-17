
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from datetime import datetime
from matplotlib import cm
import matplotlib.colors as mcolors


        ### Parameters for loading data ###
MAXTOP=5
MAXBOT=12
NE = 11
NR = 4
chunk_size = 1000000
directory = r'C:\Users\Nick\PycharmProjects\Researchcode (1) (1)\InferedTrajectoriesMCBRAIN'
savdir = r'C:\Users\Nick\PycharmProjects\Researchcode (1) (1)\CurrentProjects\Perception_TrajectoryCalc\Traj_Imgs'
def load_and_concatenate(directory, prefix, num_chunks):
    concatenated_array = []
    for i in range(num_chunks):
        chunk_file = os.path.join(directory, f'{prefix}_chunk{i}.npy')
        chunk_array = np.load(chunk_file)
        concatenated_array.append(chunk_array)

    concatenated_array = np.concatenate(concatenated_array)
    print(len(concatenated_array))
    if total_length<len(concatenated_array):
        print('ye')
        concatenated_array= concatenated_array[:total_length]
        print(len(concatenated_array))
    return concatenated_array
def pull_traj_givenSi(longtraj, ntraj, Ltraj, Si):
    ai, bi, ci, di = Si
    longa, longb, longc, longd = longtraj
    x_list,y_list,xb_list,yb_list = [],[],[],[]

    # Convert to numpy arrays for faster computation
    longa, longb, longc, longd = map(np.array, [longa, longb, longc, longd])

    # Find indices where all conditions are met
    indices = np.where((longa == ai) & (longb == bi) & (longc == ci) & (longd == di))

    # Loop over the indices and extract the trajectories
    for i in indices[0][:ntraj]:
        sample = longa[i:i+Ltraj], longb[i:i+Ltraj], longc[i:i+Ltraj], longd[i:i+Ltraj]
        x, y, xb, yb = shift_toXY(sample, NE, NR)
        x_list.append(x)
        y_list.append(y)
        xb_list.append(xb)
        yb_list.append(yb)
    return x_list, y_list, xb_list, yb_list
def shift_toXY(Traj, NE,NR):
    A,B,C,D = Traj
    N = len(A)
    x_list, y_list, xb_list, yb_list = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
    x_list[:]= C[:]-D[:]
    y_list[:]= A[:]-B[:]
    xb_list[:]=C[:]+D[:]
    yb_list[:]=A[:]+B[:]

    x_list = np.round(x_list)/NE
    xb_list = np.round(xb_list)/NE
    y_list = np.round(y_list) / NR
    yb_list = np.round(yb_list) / NR

    return x_list,y_list, xb_list,yb_list


### Jochen's Functions ###
def CountVisitsInXYXb( X_i, Y_i, Xb_i, NE, NR ):

             # convert to integer positions
    nX_i  = np.round( X_i  * NE )
    nY_i  = np.round( Y_i  * NR )
    nXb_i = np.round( Xb_i * NE )

             # list of unique values
    uX = np.unique(nX_i)
    uY = np.unique(nY_i)
    uXb = np.unique(nXb_i)

             # retrieve original coordinates
    nE_i  = (nX_i + nXb_i)/2
    nEp_i = (nX_i - nXb_i)/2

             # list of unique values
    uE    = np.unique(nE_i)
    uEp   = np.unique(nEp_i)

             # print ranges, as they are needed for the next figure
    print( (min(uX), max(uX)), 'uxrange' )
    print( (min(uY), max(uY)),'uyrange' )
    print( (min(uXb), max(uXb)) ,'uxb range')

    print( (min(uE), max(uE)) )
    print( (min(uEp), max(uEp)) )

             # Number of unique elements
    M = len(uX)
    N = len(uY)
    P = len(uXb)

             # grids of unique combinations
    nX, nY, nXb = np.meshgrid(uX, uY, uXb, indexing='ij')

    countXYXb = np.zeros((M, N, P))

    for m in range(M):
        for n in range(N):
            for p in range(P):
                       # Use logical AND to create a boolean mask and sum the occurrences
                countXYXb[m, n, p] = np.sum((nX_i == nX[m, n, p]) & (nY_i == nY[m, n, p]) & (nXb_i == nXb[m, n, p]))

    return M, N, P, uX, uY, uXb, countXYXb
def CountVisitsInXYYb( X_i, Y_i, Yb_i, NE, NR ):

             # convert to integer positions
    nX_i  = np.round( X_i  * NE )
    nY_i  = np.round( Y_i  * NR )
    nYb_i = np.round( Yb_i * NR )

             # list of unique values
    uX = np.unique(nX_i)
    uY = np.unique(nY_i)
    uYb = np.unique(nYb_i)

             # Number of unique elements
    M = len(uX)
    N = len(uY)
    P = len(uYb)

             # grids of unique combinations
    nX, nY, nYb = np.meshgrid(uX, uY, uYb, indexing='ij')

    countXYYb = np.zeros((M, N, P))

    for m in range(M):
        for n in range(N):
            for p in range(P):
                       # Use logical AND to create a boolean mask and sum the occurrences
                countXYYb[m, n, p] = np.sum((nX_i == nX[m, n, p]) & (nY_i == nY[m, n, p]) & (nYb_i == nYb[m, n, p]))

    return M, N, P, uX, uY, uYb, countXYYb
# assign colorindex based on angle in X-Y plane
def GetColorIndex( X, Y, num_colors ):
    angle = np.arctan2( Y, X)
    cix   = round( (angle + np.pi) / (2 * np.pi) * (num_colors - 1) )
    return cix
# Define an interactive plot
def plot_trajectory_density(XYtrajs,I):
    """(3D trajectory density)"""

    X_i,Y_i,Xb_i,Yb_i = XYtrajs

    # count trajectory visits in X-Y-Xb space
    (M, N, P, uX, uY, uXb, countXYXb) = CountVisitsInXYXb( X_i, Y_i, Xb_i, NE, NR )

    # Create a figure
    fig = plt.figure(figsize=(10, 6))
    # Make colormap and prepare scaling
    clrmp = plt.get_cmap('jet')
    num_colors = 256
    clrmp_array = clrmp(np.linspace(0, 1, num_colors))
    maxuX = np.max(uX)
    maxuY = np.max(uY)

    # First subplot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for m in range(0,M):
        nX = uX[m]
        for n in range(0,N):
             nY = uY[n]
             cix = GetColorIndex( nX/maxuX, nY/maxuY, num_colors )
             for p in range(0,P):
                 nXb = uXb[p]
                 cnt = countXYXb[m,n,p]
                 if cnt > 0:
                     mksz = 2*np.log( cnt )+1
                     ax1.plot(nX/NE, nY/NR, nXb/NE, marker='o', color=clrmp_array[cix,:], markersize=mksz)

    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_zlabel('X bar', fontsize=12)
    ax1.set_xlim([-0.35, 0.35])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([-0.1, 0.9])
    ax1.set_title('Trajectory density', fontsize=13)

    # count trajectory visits in X-Y-Yb space
    (M, N, P, uX, uY, uYb, countXYYb) = CountVisitsInXYYb( X_i, Y_i, Yb_i, NE, NR )

    # Second subplot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for m in range(0,M):
        nX = uX[m]
        for n in range(0,N):
             nY = uY[n]
             cix = GetColorIndex( nX/maxuX, nY/maxuY, num_colors )
             for p in range(0,P):
                 nYb = uYb[p]
                 cnt = countXYYb[m,n,p]
                 if cnt > 0:
                     mksz = 2*np.log( cnt )+1
                     ax2.plot(nX/NE, nY/NR, nYb/NR, marker='o', color=clrmp_array[cix,:], markersize=mksz)

    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('Y bar', fontsize=12)
    ax2.set_xlim([-0.35, 0.35])
    ax2.set_ylim([-1.1, 1.1])
    ax2.set_zlim([-0.1, 2.1])
    ax2.set_title('Trajectory density', fontsize=13)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.2)
    #plt.tight_layout()

    # Display the plots
    today = datetime.today().strftime('%Y-%m-%d')

    filename = f"I_{I}_MCalPerceptionTrajDensity_{today}.png"
    fullpath = os.path.join(savdir, filename)

    plt.savefig(fullpath)
    print(f'plot saved to {fullpath}')
    plt.show()
    return uX,uY,uXb

def plot_average_trajectory_flow(Ltraj,Ntraj,fulltraj,xyxb_space,I):
    """(3D trajectory flow)"""

    Ni = Ltraj
             # taken from previous figure
    uX,uY,uXb = xyxb_space

    M = len(uX) ## (a-b) space [-NR, NR]
    N = len(uY) ## (c-d) space [-NE, NE]
    P = len(uXb) ## (a+b) space [0, 2NE]

    # allocate nan arrays (they won't be filled completely)
    X_mnpi   = np.full( (M, N, P, Ni), np.nan )
    Y_mnpi   = np.full( (M, N, P, Ni), np.nan )

    Xb_mnpi  = np.full( (M, N, P, Ni), np.nan )
    Yb_mnpi  = np.full( (M, N, P, Ni), np.nan )

    # loop over nodes of state space (only nodes traversed by trajectories)
    #want n loop just to be 0 and 1
    for n in range(0,2):

        #loop through Y [Difference in upper layer]


        if n == 0:
            nY = min(uY) #cmax
            R0 = 1 * NR
            R0p = 0

        else:
            nY = max(uY) #dmax
            R0  = 0
            R0p = 1*NR

        for m in range(0,M):
            nX = uX[m]

            print( m, n )

            for p in range(0,P):
                nXb = uXb[p]

                        # intial states at lower level
                nE0  = ( nX + nXb ) / 2
                nE0p = (-nX + nXb ) / 2

                E0  = nE0   / NE
                E0p = nE0p  / NE

                        # restrict to starting points visited by trajectories
                if ( (nE0 >= 0) & (nE0p >= 0) & (nE0 <= 10) & (nE0p <= 10) & (np.remainder(nE0, 1) == 0) & (np.remainder(nE0p, 1) == 0) ):
                    E0*=NE
                    E0p*=NE
                    print(E0,E0p,R0,R0p,'starting point')
                    [X_ni, Y_ni, Xb_ni, Yb_ni] = pull_traj_givenSi( fulltraj, Ntraj, Ltraj, (R0,R0p,E0,E0p))

                    X_mnpi[m,n,p,:]    = np.squeeze( np.mean(X_ni, axis=0))
                    Y_mnpi[m,n,p,:]    = np.squeeze( np.mean(Y_ni, axis=0))
                    Xb_mnpi[m,n,p,:]   = np.squeeze( np.mean(Xb_ni, axis=0))
                    Yb_mnpi[m,n,p,:]   = np.squeeze( np.mean(Yb_ni, axis=0))

    # Create a figure and a grid of subplots (1 rows, 1 columns)
    fig = plt.figure(figsize=(6, 6))


    # Get colormap and scaling
    clrmp = plt.get_cmap('coolwarm')
    #num_colors = 256
    num_colors = 256
    clrmp_array = clrmp(np.linspace(0, 1, num_colors))
    maxuX = np.max(uX)
    maxuY = np.max(uY)

    # First subplot
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    #set limits first so that arrows are scaled correctly
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([0.0, 0.85])

    # Now get the axis limits to calculate the scale factors
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()

    # Calculate the data range for each axis
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]

    # Scale factors based on axis ranges
    x_scale = 1 / x_range
    y_scale = 1 / y_range
    z_scale = 1 / z_range


    for m in range(0,M):
        nX = uX[m]
        for n in range(0,2):
            if n == 0:
             nY = min(uY)
            else:nY = max(uY)
            cix = GetColorIndex( nX/maxuX, nY/maxuY, num_colors )
            for p in range(0,P):

                X_i   = np.squeeze( X_mnpi[m,n,p,:] )
                Y_i   = np.squeeze( Y_mnpi[m,n,p,:] )
                Xb_i  = np.squeeze( Xb_mnpi[m,n,p,:] )

                if np.sum( ~np.isnan(X_i) )>0:

                    num_points = len(X_i)

                    ax1.scatter(X_i, Y_i, Xb_i, marker='.', color=clrmp_array[cix,:], linewidth=.3,zorder=1, alpha=0.85)
                    #thin lines between points plotted
                    ax1.plot(X_i, Y_i, Xb_i, color=clrmp_array[cix, :]*.6, linewidth=0.15, zorder=2,alpha=0.85)
                    #add a little arrow of the same length to show direction from starting point to second point in the same color

                    #find arrow direction, and normalize according to scale size

                    for i in range(0, num_points - 2, int(num_points/3)):  # Add an arrow every 10th point
                        direction = np.array([X_i[i+1] - X_i[i], Y_i[i+1] - Y_i[i], Xb_i[i+1] - Xb_i[i]])
                        direction = np.array([direction[0] * x_scale, direction[1] * y_scale, direction[2] * z_scale])
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction /= norm
                        length = .10
                        arrow_V = length * direction
                        if abs(Y_i[i])<= 0.9:

                            ax1.quiver(X_i[i], Y_i[i], Xb_i[i], arrow_V[0]/x_scale, arrow_V[1]/y_scale, arrow_V[2]/z_scale, color=clrmp_array[cix, :], linewidth=2.2, zorder=5)
                            #ax1.quiver(X_i[i], Y_i[i], Xb_i[i], arrow_V[0]/x_scale, arrow_V[1]/y_scale, arrow_V[2]/z_scale, color='black', linewidth=3, zorder=5)

    ax1.set_xlabel('X', fontsize=14)
    ax1.set_ylabel('Y', fontsize=14)
    ax1.set_zlabel('X bar', fontsize=14)

    ax1.set_title('Trajectory flow', fontsize=18)

    # Adjust layout to prevent overlap
    today = datetime.today().strftime('%Y-%m-%d')

    filename = f"I_{I}_MCalPerceptionAvgFlow_{today}_L={num_points}.png"
    fullpath = os.path.join(savdir, filename)

    plt.savefig(fullpath)
    print(f'plot saved to {fullpath}')
    plt.show()


if __name__ == '__main__':
            # load raw data 0-N integers#
    Is = ['1','6875','375','0625']
    I_test = Is[3]

    ### LOAD FULL DATA ###
    total_length = 50_000_001
    chunk_size = 1_000_000
    # Calculate the number of chunks needed #
    num_chunks = total_length // chunk_size
    if total_length % chunk_size != 0:
        num_chunks += 1

    inf_trajA = load_and_concatenate(directory, 'JochenI_' + I_test + 'dt_.001HseedA', num_chunks)
    inf_trajB = load_and_concatenate(directory, 'JochenI_' + I_test + 'dt_.001HseedB', num_chunks)
    inf_trajC = load_and_concatenate(directory, 'JochenI_' + I_test + 'dt_.001HseedC', num_chunks)
    inf_trajD = load_and_concatenate(directory, 'JochenI_' + I_test + 'dt_.001HseedD', num_chunks)

    Lsample = 30_000
    small_trajA = inf_trajA[:Lsample]
    small_trajB = inf_trajB[:Lsample]
    small_trajC = inf_trajC[:Lsample]
    small_trajD = inf_trajD[:Lsample]
    ########################################

    ### GET TRAJECTORY DENSITY ###
    x, y, xb, yb = shift_toXY((small_trajA,small_trajB,small_trajC,small_trajD), NE, NR)
    ux, uy, uxb = plot_trajectory_density((x, y, xb, yb), I_test)
    ################################

    ### GET AVERAGE TRAJECTORY FLOW ###
    nSamples = 10000
    Ltraj = 25
    plot_average_trajectory_flow(Ltraj,nSamples,(inf_trajA,inf_trajB,inf_trajC,inf_trajD),(ux,uy,uxb),I_test)
    ##############################################


