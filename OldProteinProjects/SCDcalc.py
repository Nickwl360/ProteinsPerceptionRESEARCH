from typing import List
import pandas as pd

def getseq(file_path):
    df = pd.read_excel(file_path)
    sequences = df.iloc[:, 1].values.tolist()
    return sequences

##seq[0]=sv1

def getcharges(seq):
    charges = []
    # get charge array
    for letter in seq:
        if letter == 'E' or letter == 'D':
            charges.append(-1)
        elif letter == 'R' or letter == 'K':
            charges.append(1)
        elif letter == 'X':
            charges.append(-2)
        else:
            charges.append(0)
    return charges


def calcQ(seq):
    charges = getcharges(seq)
    size = len(charges)

    # calc SCD from charge array
    SUM = 0
    nloop = 0
    for m in range(2, size+1):
        nloop = 0
        for n in range(1, m):
            nloop += charges[m - 1] * charges[n - 1] * (m - n) ** (1 / 2)
        SUM += nloop
    return SUM * (1 / size)

