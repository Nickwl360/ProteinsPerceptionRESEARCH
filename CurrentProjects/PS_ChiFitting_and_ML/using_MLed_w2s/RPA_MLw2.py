from CurrentProjects.PS_ChiFitting_and_ML.RPA_MODEL_functs import findCrit, getBinodal
import CurrentProjects.PS_ChiFitting_and_ML.ML_Lili_w2s
from CurrentProjects.LLPS_rg_RPA.rgRPA_init import pH_qs
import pandas as pd
import numpy as np
import csv
import os
import time
from matplotlib import pyplot as plt
from datetime import datetime

class Protein:
    def __init__(self,name, sequence,w2,w3):
        self.name = name
        self.sequence = sequence
        self.w2 = w2
        self.w3 = w3
        self.qc ,self.q_list, self.N = self.calculate_props_fromseq(sequence)
        self.xeeSig, self.gkSig, self.ckSig = self.getSigShifts(self.N, self.q_list)
        self.L = np.arange
        self.W3_TOGGLE = 0
        self.i_vals = np.arange(0, self.N)
        self.qL = np.array(self.q_list)
        self.Q = np.sum(self.qL*self.qL)/self.N
        self.nres = 30
        self.minFrac= .7

        self.phiC = None
        self.Yc = None
        self.Ymin = None

        self.Ybin = None
        self.spinbin = None
        self.bibin= None
        self.Yspace = None


    def calculate_props_fromseq(self, sequence):
        q_list= pH_qs(sequence,ph=7)
        N = len(q_list)
        qc = abs(sum(q_list))/N
        return qc, q_list,N
    def getSigShifts(self, N, qs):

        sigSij = []
        for i in range(len(qs)):  ###abs(tau - mu)
            sigij = 0
            for j in range(len(qs) - 1):  #### tau (starting spot)
                if (j + i) <= len(qs) - 1:
                    sigij += qs[j] * qs[j + i]  #
            if i == 0:
                sigSij.append(sigij)
            if i != 0:
                sigSij.append(2 * sigij)

        sigGs = 2 * np.arange(N, 0, -1)
        sigGs[0] /= 2
        #######THIS IS FROM LIN GITHUB, COULDN'T FIGURE OUT MY SUM METHOD

        mlx = np.kron(qs, np.ones(N)).reshape((N, N))
        sigSi = np.array([np.sum(mlx.diagonal(n) + mlx.diagonal(-n)) for n in range(N)])
        sigSi[0] /= 2

        return sigSij, sigSi, sigGs
    def getCrits(self):
        self.phiC, self.Yc = findCrit(self)
        self.Ymin= self.Yc*self.minFrac
        self.Yspace = np.logspace(0,np.log10(self.Ymin/self.Yc),num=self.nres)*self.Yc
        self.Yspace = self.Yspace[1:]
    def getCurves(self):
        if not np.isnan(self.phiC) and not np.isnan(self.Yc):
            self.spinbin, self.bibin, self.Ybin = getBinodal(self)
        else:
            # If not valid, mark spinodal and binodal curves as not available
            self.spinbin = None
            self.bibin = None
            print(f"No binodal or spinodal curve can be found for {self.name}")

    def __repr__(self):
        return (f"Name({self.name},length={self.N},charge={self.qc},w2={self.w2}")


### this method reads from a csv and creates wanted Protein objects based off of it ###
def load_proteins_fromcsv(file_path):
    df = pd.read_csv(file_path)
    proteinList = [Protein(name=row['Name'], sequence=row['Sequence'], w2=row['w2_preds_LLw302'], w3=0.2) for _,row in df.iterrows()]
    return proteinList



### a little silly, but this method takes specific input from user for determining which proteins to use for model. ###
### will create a more general purpose method but this works for now ###
def run_selected_proteins(proteinobj_list):
    for indx, protein in enumerate(proteinobj_list):
        print(f"Index: {indx},Name: {protein.name}")


    index_input = input("Enter specific protein indices to run (e.g., 1,2,6,3,7): ").strip()
    selected_indices = list(map(int, index_input.split(',')))
    selected_proteins = [proteinobj_list[idx] for idx in selected_indices]

    for protein in selected_proteins:
        run_model_onProtein(protein)

    plot_binodals(selected_proteins)
    return

def run_model_onProtein(protein):
    print(f"Processing {protein.name}:")
    print(f"- Sequence Length: {protein.N}")
    print(f"- Total Charge: {protein.qc}")
    #print(f"- Charge List: {protein.q_list}")
    print(f"- w2 Value: {protein.w2}")
    #print(f"- xeeshift: {protein.xeeSig}")
    tik = time.time()
    protein.getCrits()
    print(protein.phiC,protein.Yc,'crit found in ', (time.time()-tik), ' s \n')

    print(f'SOLVING FOR A BINODAL CURVE (Nres = {protein.nres})')
    tok = time.time()
    protein.getCurves()
    print('binodal calculated in ', (time.time()-tok), ' s \n')

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


def plot_binodals(protein_list):
    plt.figure()
    YcNorm = protein_list[0].Yc
    min_ybin = np.inf
    max_ybin = -np.inf
    max_phibin = -np.inf

    for protein in protein_list:
        if protein.bibin is not None:
            ybinNorm = protein.Ybin/YcNorm
            min_ybin = min(min_ybin, np.min(ybinNorm))
            max_ybin = max(max_ybin, np.max(ybinNorm))
            max_phibin = max(max_phibin, np.max(protein.bibin))
            #plt.plot(protein.spinbin, ybinNorm, label= protein.name)
            plt.plot(protein.bibin, ybinNorm, label= protein.name)

            #plt.plot(protein.bibin, ybinNorm, label= protein.name[11:])


    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$T^*$')
    plt.title(f'{protein_list[0].name[:6]} and Mutants')
    plt.ylim(0.95 * min_ybin, 1.05 * max_ybin)
    plt.xlim(0.0,.15)
    plt.legend(fontsize=9)
    plt.show()

    run_saver(plt,protein_list)

def run_saver(plot,proteinlist):
    save_bool = input("do you want to save this plot? (Y/N) ").strip().upper()
    if save_bool=='Y':
        today = datetime.today().strftime('%Y-%m-%d')

        filename = f"{proteinlist[0].name[:6]}&mutants_fgRPA_w2Pred_{today}.png"
        savedir = r'C:\Users\Nick\PycharmProjects\Researchcode (1) (1)\CurrentProjects\PS_ChiFitting_and_ML\using_MLed_w2s\FH_PhaseDiagrams'

        fullpath = os.path.join(savedir,filename)

        plot.savefig(fullpath)
        print(f'plot saved to {fullpath}')



if __name__ == '__main__':
    df = r'C:\Users\Nick\PycharmProjects\Researchcode (1) (1)\CurrentProjects\PS_ChiFitting_and_ML\ML_Lili_w2s\phase_sep_seqs_w2s.csv'
    proteinlist = load_proteins_fromcsv(df)

    run_selected_proteins(proteinlist)


