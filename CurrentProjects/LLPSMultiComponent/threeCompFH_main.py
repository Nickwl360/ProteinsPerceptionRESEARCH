from threeCompFH_init import initialize
from threeCompFH_functions import *

###########CHANGE ABOVE * TO NEEDED FNCTS#################

if __name__ == "__main__":
    initialize()

    bulk1_array= np.linspace(1e-6 ,.3,200)
    bulk2_array= np.linspace(1e-6 ,.3,200)
    # bulk1_array = np.linspace(.07, .15, 50)
    # bulk2_array = np.linspace(.07, .15, 50)

    denseSet, lightSet = solveBinaryExample1(bulk1_array, bulk2_array)
    pltScatter(denseSet,lightSet)
