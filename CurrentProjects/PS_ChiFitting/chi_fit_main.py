import numpy as np
import FH_PSModel_Functs as fh
from chi_fit_init import*


#would be imported better
df_seq_csat_list= [('MCM4', 140,.032, .124)]

#define variables for model fitter to use
N = df_seq_csat_list[0][1]

phiC, chiC = fh.get_critical_vals(N)
cdilpred, cdensepred = df_seq_csat_list[0][2],df_seq_csat_list[0][3]

# want to find best chi for these 2 predictions#
chi_search_space = np.linspace(-2,2,50)



def find_best_chi(T, preds):
    phi1pred, phi2pred = preds
    bestChi = 0

    for chiTest in chi_search_space:

        FH_phi1, FH_phi2 ,s1,s2 = fh.findPhisnoconst(T, phiC)

        if np.isclose(FH_phi1,phi1pred,tol) and np.isclose(FH_phi2, phi2pred, tol):
            print('WE HAVE FOUND A FH MODEL TO MATCH PREDICTION for ', df_seq_csat_list[0][0])
            bestChi = chiTest


    return bestChi

def cdense_calc(dG, cdil):
    return cdil/(np.exp(dG))
def vFrac_fromMgMl(mgml):
    return

print(np.exp(3.935),cdense_calc(-1.0578, np.exp(3.935)))