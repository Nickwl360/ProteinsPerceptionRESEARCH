import numpy as np

prog_path = r'C:\Users\Nickl\PycharmProjects\Researchcode (1) (1)\CurrentProjects\PerceptionCUDA_TrajectoryCalc\Tij_kernel.cl'
MAXTOP, MAXBOT = 5,12

#hamiltonian
epsilon1,epsilon2 = 0,0
(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) =(-6.00432578 ,-6.03486149, -3.80626702 , 5.35061176 , 6.4340547  , 3.67099913 ,8.6066445  , 0.28647123 , 2.42905138)
params = (halpha, ha, halpha - epsilon1, ha + epsilon1,hgamma,hc,hgamma-epsilon2,hc +epsilon2, kcoop, kcomp,kdu,kud,kx)

#initial conditions
ll1,ll2 = 6,3

#trajectory stuff
dt = .001
