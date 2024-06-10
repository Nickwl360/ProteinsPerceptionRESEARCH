import numpy as np
from matplotlib import pyplot as plt

def EquilibriumMatrix(Pnorm,statesize,bigshape, smallshape):
    Pfix = np.reshape(Pnorm, bigshape)
    Plarge = np.linalg.matrix_power(Pfix, 100000)
    Plarge = Plarge.transpose()
    evalues, evectors = np.linalg.eig(Plarge)
    threshold = 1e-6
    indices_close_to_1 = [i for i, eigenvalue in enumerate(evalues) if abs(eigenvalue - 1) < threshold]
    eigenvectors_close_to_1 = [evectors[:, i] for i in indices_close_to_1]
    states = np.zeros(statesize,np.float64)
    for i in indices_close_to_1:
        if len(indices_close_to_1)>1:
            toadd = eigenvectors_close_to_1[i]
            toadd = np.real(toadd)
            states += toadd
        if len(indices_close_to_1)==1:
            state = eigenvectors_close_to_1[i]
            state = np.real(state)
            state = state.reshape(smallshape)
            state = state.transpose()

    if len(indices_close_to_1)==1:
        return state
    else:
        states[:]/=states.sum()
        states = states.reshape(smallshape)
        states = states.transpose()
        return states


