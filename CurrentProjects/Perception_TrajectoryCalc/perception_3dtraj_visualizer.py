
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


        ### Parameters for loading data ###
total_length = 100_001
chunk_size = 1_000_000
        # Calculate the number of chunks needed #
num_chunks = total_length // chunk_size
if total_length % chunk_size != 0:
    num_chunks += 1

MAXTOP=5
MAXBOT=12
NE = 11
NR = 4
chunk_size = 1000000
directory = r'C:\Users\Nickl\PycharmProjects\Researchcode (1) (1)\InferedTrajectoriesMCBRAIN'
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
    print( (min(uX), max(uX)) )
    print( (min(uY), max(uY)) )
    print( (min(uXb), max(uXb)) )

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
                countXYXb[m, n, p] = np.sum((nX_i == nX[m, n, p]) & (nY_i == nY[m, n, p]) & (nXb_i == nXb[m, n, p]));

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
def plot_trajectory_density(XYtrajs):
    """(3D trajectory density)"""

    X_i,Y_i,Xb_i,Yb_i = XYtrajs

    # count trajectory visits in X-Y-Xb space
    (M, N, P, uX, uY, uXb, countXYXb) = CountVisitsInXYXb( X_i, Y_i, Xb_i, NE, NR )

    # Create a figure
    fig = plt.figure(figsize=(6, 12))
    # Make colormap and prepare scaling
    clrmp = plt.get_cmap('jet')
    num_colors = 256
    clrmp_array = clrmp(np.linspace(0, 1, num_colors))
    maxuX = np.max(uX)
    maxuY = np.max(uY)

    # First subplot
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
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

    ax1.set_xlabel('X', fontsize=14)
    ax1.set_ylabel('Y', fontsize=14)
    ax1.set_zlabel('X bar', fontsize=14)
    ax1.set_xlim([-0.35, 0.35])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([0.0, 1.1])
    ax1.set_title('Trajectory density', fontsize=18)

    # count trajectory visits in X-Y-Yb space
    (M, N, P, uX, uY, uYb, countXYYb) = CountVisitsInXYYb( X_i, Y_i, Yb_i, NE, NR )

    # Second subplot
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
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

    ax2.set_xlabel('X', fontsize=14)
    ax2.set_ylabel('Y', fontsize=14)
    ax2.set_zlabel('Y bar', fontsize=14)
    ax2.set_xlim([-0.35, 0.35])
    ax2.set_ylim([-1.1, 1.1])
    ax2.set_zlim([0.0, 2.1])
    ax2.set_title('Trajectory density', fontsize=18)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()


if __name__ == '__main__':
            # load raw data 0-N integers#
    Is = ['1','6875','375','0625']

    I_test = Is[3]

    inf_trajA = load_and_concatenate(directory, 'JochenI_'+I_test+'dt_.001HseedA', num_chunks)
    inf_trajB = load_and_concatenate(directory, 'JochenI_'+I_test+'dt_.001HseedB', num_chunks)
    inf_trajC = load_and_concatenate(directory, 'JochenI_'+I_test+'dt_.001HseedC', num_chunks)
    inf_trajD = load_and_concatenate(directory, 'JochenI_'+I_test+'dt_.001HseedD', num_chunks)

    x, y, xb, yb = shift_toXY((inf_trajA, inf_trajB, inf_trajC, inf_trajD), NE, NR)
    plot_trajectory_density((x,y,xb,yb))


