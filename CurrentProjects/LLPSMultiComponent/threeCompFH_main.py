from threeCompFH_init import initialize
from threeCompFH_functions import *

###########CHANGE ABOVE * TO NEEDED FNCTS#################

def generateBulkInputMatrix(n_points, max_value):
    # Create initial linspace arrays
    bulk1_array = np.linspace(1e-6, max_value, n_points)

    valid_pairs = []

    for phi1 in bulk1_array:
        for phi2 in np.linspace(1e-6, max_value - phi1, n_points):
            if phi1 + phi2 <= 1:
                valid_pairs.append([phi1, phi2])
            else:
                break

    # Convert list to a numpy array
    valid_pairs_array = np.array(valid_pairs)

    return valid_pairs_array


if __name__ == "__main__":
    initialize()
    bulkMatrix = generateBulkInputMatrix(n_points=300, max_value=(.25 - epsilon))
    pair1 = [.1,.1]
    bulkMatrix = np.vstack(([pair1], bulkMatrix))

    denseSet, lightSet, triA,triB,triC = solveBinaryExample1(bulkMatrix)
    tri=(triA,triB,triC)
    pltScatter(denseSet,lightSet,tri,20)
