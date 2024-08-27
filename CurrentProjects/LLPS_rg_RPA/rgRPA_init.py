import numpy as np
from OldProteinProjects.SCDcalc import *

def pH_qs(seq, ph):
    charges = []
    # get charge array
    for letter in seq:
        if letter == 'E' or letter == 'D':
            if letter == 'E':
                q = -1*(10**(-1*(4.15- ph)))/(1+ 10**(-1*(4.15-ph)))
            elif letter == 'D':
                q = -1 * (10 ** (-1*(3.71 - ph))) / (1 + 10 ** (-1*(3.71 - ph)))
            charges.append(q)
        elif letter == 'R' or letter == 'K' or letter == 'H':
            if letter == 'R':
                q = (10**(12.1- ph))/(1 + 10**(12.1-ph))
            elif letter == 'K':
                q = (10**(10.67- ph))/(1 + 10**(10.67-ph))
            elif letter =='H':
                q = (10**(6.04-ph))/(1+10**(6.04-ph))
            charges.append(q)
        else:
            charges.append(0)
    return charges

###########CONSTANTS##########################
phiS = .0
ph = 7
scale_init= .005
scale_final= .01
epsilon = 1e-12

#################PICK SEQUENCE/GET RELEVANTQUANTITIES###################################################
seqs = getseq('../../OldProteinProjects/SCDtests.xlsx')
ddx4n1 = 'MGDEDWEAEINPHMSSYVPIFEKDRYSGENGDNFNRTPASSSEMDDGPSRRDHFMKSGFASGRNFGNRDAGECNKRDNTSTMGGFGVGKSFGNRGFSNSRFEDGDSSGFWRESSNDCEDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFGSRRPVLSGTGNGDTSQSRSGSGSERGGYKGLNEEVITGSGKNSWKSEAEGGES'
ddx4n1CS = 'MGDRDWRAEINPHMSSYVPIFEKDRYSGENGRNFNDTPASSSEMRDGPSERDHFMKSGFASGDNFGNRDAGKCNERDNTSTMGGFGVGKSFGNEGFSNSRFERGDSSGFWRESSNDCRDNPTRNDGFSDRGGYEKGNNSEASGPYERGGRGSFDGCRGGFGLGSPNNRLDPRECMQRTGGLFGSDRPVLSGTGNGDTSQSRSGSGSERGGYKGLNEKVITGSGENSWKSEARGGES'
IP5 = 'HAQGTFTSDKSKYLDERAAQDFVQWLLDGGPSSGAPPPS'
seq_of_interest = ddx4n1CS
qs = pH_qs(seq_of_interest,ph)
#qs = getcharges(seq_of_interest) #regular way
N = len(qs)
qc = abs(sum(qs))/N

### LIST OF KNOWN CRIT POINTS FOR QUICKER TESTING ###
rg_phiC_list = (['IP5', .020014], ['ddx4n1',.015],['ddx4n1CS',.01983037])
phiC_test = 0
for name, pc in enumerate(rg_phiC_list):
    if name == seq_of_interest: phiC_test=pc





