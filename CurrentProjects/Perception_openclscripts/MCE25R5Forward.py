from MCBrainE25R5 import *
import numpy as np

Tmax = 10_000

def simulate(params,total_time,n0):
    NA, NB,NC,ND = [n0[0]],[n0[1]],[n0[2]],[n0[3]]
    #CALC P(Ns,Params):
    #SAVE P(N1)
    #FIND NEXT STATE, CHECK IF ALREADTY CALCD
    #LOOPIT

    return NA,NB,NC,ND




if __name__ == "__main__":
    epsilon1, epsilon2 = 0,0
    #(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx)= (-8.96733557, -7.73231853, -6.01935508 ,-0.99322105,  4.7228139 ,  1.98114397 ,6.05944224 , 0.29747507 , 1.53067954)

    #params = (halpha, ha, halpha - epsilon1, ha + epsilon1, hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)
    Nstart=(0,0,0,0)
    trajA,trajB,trajC,trajD = simulate(params,Tmax, Nstart)