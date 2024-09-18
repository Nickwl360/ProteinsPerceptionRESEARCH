import numpy as np
import FH_PSModel_Functs as fh
from chi_fit_init import*


cdilpred, cdensepred = df_seq_csat_list[0][2],df_seq_csat_list[0][3]
preds =(cdilpred,cdensepred)
# want to find the best chi for these 2 predictions#
chi_search_space = np.linspace(.3,1,5000)

def find_best_chi(T, preds ):
    phi1pred, phi2pred = preds
    bestChi = 0

    for chiTest in chi_search_space:
        phiC, tC = fh.get_critical_vals(N, T, chiTest)
        #print(phiC, tC, 'critical values')
        if tC>T:

            FH_phi1, FH_phi2 ,s1,s2 = fh.findPhisnoconst(T, chiTest, phiC)
            print(chiTest,FH_phi1,FH_phi2)

            if np.isclose(FH_phi1,phi1pred,tol) and np.isclose(FH_phi2, phi2pred, tol):
                print('WE HAVE FOUND A FH MODEL TO MATCH PREDICTION for ', df_seq_csat_list[0][0])
                bestChi = chiTest
                break
            elif np.isclose(FH_phi1,phi1pred,tol) and not (np.isclose(FH_phi2,phi2pred,tol)):
                print('ONLY CSAT MATCHES, NOT DENSE PHASE', FH_phi1,phi1pred, FH_phi2,phi2pred)
                bestChi =chiTest
                break
        else:pass

    return bestChi


#print(find_best_chi(T, preds))


