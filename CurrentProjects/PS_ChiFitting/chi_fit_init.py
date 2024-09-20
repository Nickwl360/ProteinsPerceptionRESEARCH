import numpy as np
import pandas as pd

#experimental constants
e = 1.60217e-19
e0 = 8.854e-12
er = 80
kb = 1.380649e-23
b = 3.8e-10  #bond length
MWavg = 110 #dalton (avg molecular weight per aminoacid

#constants
epsilon = 1e-15
Tmin = 0.1
w3 = .2

#### data processing ###

def cdense_calc(dG, cdil):
    return cdil/(np.exp(dG))

def vFrac_fromMgMl(mgml):
    phi = 6.022e-4 / MWavg * (b*1e10) ** 3 * mgml
    return phi
def vFrac_frommM(mM,N):
    phi = 6.022e-7*N*(b*1e10)**3 * mM
    return phi

csat_exp = np.exp(3.935)
cdense_exp = cdense_calc(-1.0578, csat_exp)
#
# csat_vfrac = vFrac_fromMgMl(csat_exp)
# cdense_vfrac = vFrac_fromMgMl(cdense_exp)
#
csat_vfrac = vFrac_frommM(.102,135)
cdense_vfrac = vFrac_frommM(26.03, 135)
#
# csat_vfrac = .000102
# cdense_vfrac = .026031
# print(csat_vfrac,cdense_vfrac)


df_seq_csat_list= [('MCM4', 137, csat_vfrac, cdense_vfrac)]

seqData = pd.DataFrame(df_seq_csat_list, columns=['seq_name','N','csat','cdense'])
#save as: name, N, cdil, cdense

#convert to vfrac

## toggles
seq_of_interest = 'MCM4'

## function toggles
omega3toggle = 0

#define variables for model fitter to use
N = df_seq_csat_list[0][1]
resolution = 1e-3
TTest = 293.15-273.15

