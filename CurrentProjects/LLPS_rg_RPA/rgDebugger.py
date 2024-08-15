import numpy as np
from rgRPA_allfns import FBINODAL
from scipy import integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from rgRPA_init import *
from scipy.optimize import brenth

test1= np.linspace(epsilon, .2, 100)
test2= np.linspace(epsilon, .2, 100)

def getEnergyLandscape(phi1ray, phi2ray ):
    Phi1, Phi2 = np.meshgrid(phi1ray, phi2ray)
    Energy = np.zeros_like(Phi1)
    for i in range(Phi1.shape[0]):
        print(i)
        for j in range(Phi1.shape[1]):
            Energy[i, j] = FBINODAL((Phi1[i, j], Phi2[i, j]),.240,.02)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Phi1, Phi2, Energy, cmap='viridis')

    # Labels and title
    ax.set_xlabel('Phi1')
    ax.set_ylabel('Phi2')
    ax.set_zlabel('Energy')
    ax.set_title('Energy Landscape')

    plt.show()
    return

getEnergyLandscape(test1,test2)
