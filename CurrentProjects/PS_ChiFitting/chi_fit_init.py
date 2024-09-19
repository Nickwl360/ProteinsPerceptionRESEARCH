import numpy as np
import pandas as pd

#experimental bs
e = 1.60217e-19
e0 = 8.854e-12
er = 29.5 # or 80
kb = 1.380649e-23
b = 3.8e-10  #bond length


#constants
epsilon = 1e-15
MWavg = 110 #dalton (avg molecular weight per aminoacid

tol = 1e-2


def cdense_calc(dG, cdil):
    return cdil/(np.exp(dG))

def vFrac_fromMgMl(mgml):
    phi = 6.022e-4 / MWavg * b ** 3 * mgml
    return phi

csat_exp = np.exp(3.935)
cdense_exp = cdense_calc(-1.0578, csat_exp)

csat_vfrac = vFrac_fromMgMl(csat_exp)
cdense_vfrac = vFrac_fromMgMl(cdense_exp)

df_seq_csat_list= [('MCM4', 32, csat_vfrac, cdense_vfrac)]

#steps:
seqData = pd.DataFrame(df_seq_csat_list, columns=['seq_name','N','csat','cdense'])
#save as: name, N, cdil, cdense

#convert to vfrac


seq_of_interest = 'MCM4'

#define variables for model fitter to use
N = df_seq_csat_list[0][1]
resolution = .01
TTest = 293-273.15

Tmin = epsilon
