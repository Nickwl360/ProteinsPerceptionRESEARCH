import numpy as np
import FH_PSModel_Functs as fh
from chi_fit_init import*
from scipy.optimize import minimize_scalar


cdilpred, cdensepred = df_seq_csat_list[0][2],df_seq_csat_list[0][3]
preds =(cdilpred,cdensepred)

def Tstar_to_Tcelsius(Tstar):
    T_kelvin = Tstar*e*e/(4*np.pi*e0*er*kb*b)
    T_celsius = T_kelvin - 273.15
    return T_celsius
def Tcelsius_to_Tstar(Tcelsius):
    Tkelvin = Tcelsius + 273.15
    Tstar = Tkelvin*(4*np.pi*e0*er*kb*b)/(e*e)
    return Tstar

def error_function(chi, T, preds):
    phi1pred, phi2pred = preds
    phiC, tC = fh.get_critical_vals(N, T, chi)

    if tC > T:
        FH_phi1, FH_phi2, s1, s2 = fh.findPhisnoconst(T, chi, phiC)

        # Calculate the squared error for both phases (phi1 and phi2)
        error = (FH_phi1 - phi1pred) ** 2 + (FH_phi2 - phi2pred) ** 2
        return error
    else:
        # If tC < T, return a high error so this chi is avoided
        return np.inf


def find_best_chi(T, preds):
    result = minimize_scalar(error_function, bounds=(0, 3), args=(T, preds), method='bounded')

    if result.success:
        best_chi = result.x
        print(f"Best chi found: {best_chi}")
        return best_chi
    else:
        raise ValueError("Optimization failed to converge")

T = Tcelsius_to_Tstar(Tcelsius=TTest)
best_chi = find_best_chi(T, preds)

print(f"Best chi for the given data is: {best_chi}")

