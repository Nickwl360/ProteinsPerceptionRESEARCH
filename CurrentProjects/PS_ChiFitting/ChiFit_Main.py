import numpy as np

###constants##############
epsilon = 1e-7
resolution = .001
NA = 10
NB = 1

def cdense_calc(dG, cdil):
    return cdil/(np.exp(dG))
def vFrac_fromMgMl(mgml):
    return

print(np.exp(3.935),cdense_calc(-1.0578, np.exp(3.935)))