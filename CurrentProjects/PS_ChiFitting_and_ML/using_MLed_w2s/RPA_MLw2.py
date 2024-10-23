from CurrentProjects.PS_ChiFitting_and_ML.RPA_MODEL_functs import findCrit, getBinodal
from CurrentProjects.LLPS_rg_RPA.rgRPA_init import pH_qs
import pandas as pd
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from datetime import datetime
import rg_RPA_model as rg

class Protein:
    def __init__(self,name, sequence,w2,w3,rg,phiS=None):
        self.name = name
        self.sequence = sequence
        self.w2 = w2
        self.w3 = w3
        self.phiS = 0 if phiS == None else phiS

        self.phiS = phiS
        self.qc ,self.q_list, self.N = self.calculate_props_fromseq(sequence)
        self.xeeSig, self.gkSig, self.ckSig = self.getSigShifts(self.N, self.q_list)
        self.L = np.arange
        self.W3_TOGGLE = 1
        self.i_vals = np.arange(0, self.N)
        self.qL = np.array(self.q_list)
        self.Q = np.sum(self.qL*self.qL)/self.N

        self.nres = 15
        self.minFrac= .85

        self.phiC = None
        self.Yc = None
        self.Ymin = None

        self.Ybin = None
        self.spinbin = None
        self.bibin= None
        self.Yspace = None

        self.rg = rg
        self.epsC = .69
        self.crowding_toggle = 1


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
        if self.rg == 1:
            self.phiC, self.Yc = rg.findCrit(self)
        else:
            self.phiC, self.Yc = findCrit(self)

        self.Ymin= self.Yc*self.minFrac
        self.Yspace = np.logspace(0,np.log10(self.Ymin/self.Yc),num=self.nres)*self.Yc
        self.Yspace = self.Yspace[1:]
    def getCurves(self):
        if not np.isnan(self.phiC) and not np.isnan(self.Yc):
            if self.rg == 1:
                self.spinbin, self.bibin, self.Ybin = rg.getBinodal(self)
            else:
                self.spinbin, self.bibin, self.Ybin = getBinodal(self)
        else:
            # If not valid, mark spinodal and binodal curves as not available
            self.spinbin = None
            self.bibin = None
            print(f"No binodal or spinodal curve can be found for {self.name}")

    def __repr__(self):
        return (f"Name({self.name},length={self.N},charge={self.qc},w2={self.w2}")


### this method reads from a csv and creates wanted Protein objects based off of it ###
def load_proteins_fromcsv(file_path,rg,phiS):
    df = pd.read_csv(file_path)
    proteinList = [Protein(name=row['Name'], sequence=row['Sequence'], w2=row['w2_preds_LLw302'], w3=0.2, rg=rg,phiS=phiS) for _,row in df.iterrows()]
    return proteinList
def convPhiSto_mM(phiS):
    return phiS/(6.022e-7*(3.8)**3)
def convmMtoPhiS(mM):
    return mM*(6.022e-7*(3.8)**3)


### a little silly, but this method takes specific input from user for determining which proteins to use for model. ###
### will create a more general purpose method but this works for now ###
def select_andrun_proteins(proteinobj_list):
    for indx, protein in enumerate(proteinobj_list):
        print(f"Index: {indx},Name: {protein.name}")

    index_input = input("Enter specific protein indices to run (e.g., 1,2,6,3,7): ").strip()
    selected_indices = list(map(int, index_input.split(',')))
    selected_proteins = [proteinobj_list[idx] for idx in selected_indices]

    for protein in selected_proteins:
        run_model_onProtein(protein)

    ycNorm = selected_proteins[0].Yc
    #print normalized crit list
    for protein in selected_proteins:
        print(f"Normalized Crit Value for {protein.name}: {protein.Yc/ycNorm}")

    #perhaps add a data saver here

    plot_binodals(selected_proteins,phiS=0)
    return
def select_andrun_proteins_withsalt(proteinobj_list, phiSchoices):
    for indx, protein in enumerate(proteinobj_list):
        print(f"Index: {indx},Name: {protein.name}")

    index_input = input("Enter specific protein indices to run (e.g., 1,2,6,3,7): ").strip()
    selected_indices = list(map(int, index_input.split(',')))
    selected_proteins = [proteinobj_list[idx] for idx in selected_indices]
    phiS_selected_proteins = []

    #phiSchoices will be a list of protein names and phiS values, because some proteins have multiple phiS values
    #create duplicate proteins that have the same name but different phiS values
    for name,phiS in phiSchoices:
        for protein in selected_proteins:
            if protein.name == name:
                new_protein = Protein(name=f"{protein.name}", sequence=protein.sequence, w2=protein.w2, w3=protein.w3, rg=protein.rg, phiS=phiS)
                phiS_selected_proteins.append(new_protein)

    for protein in phiS_selected_proteins:
        run_model_onProtein(protein)

    ycNorm = phiS_selected_proteins[0].Yc
    #print normalized crit list
    for protein in phiS_selected_proteins:
        print(f"Normalized Crit Value for {protein.name}: {protein.Yc/ycNorm}")

    plot_binodals(phiS_selected_proteins,phiS=1)
    return

def run_model_onProtein(protein):
    print(f"Processing {protein.name}:")
    print(f"- Sequence Length: {protein.N}")
    print(f"- Total Charge: {protein.qc}")
    #print(f"- Charge List: {protein.q_list}")
    print(f"- w2 Value: {protein.w2}")
    print(f"- phiS Value: {protein.phiS}")
    print(f"- MODEL: fg = 0, rg =1 : {protein.rg}")
    #print(f"- xeeshift: {protein.xeeSig}")
    if protein.crowding_toggle == 1:
        print(f"- Crowding Toggle: ON, epsC = {protein.epsC} kT")
    tik = time.time()
    protein.getCrits()
    print(protein.phiC,protein.Yc,'crit found in ', (time.time()-tik), ' s \n')


    print(f'SOLVING FOR A BINODAL CURVE (Nres = {protein.nres})')
    tok = time.time()
    protein.getCurves()
    print('binodal calculated in ', (time.time()-tok), ' s \n')

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


def plot_binodals(protein_list,phiS):
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
            if phiS==1:
               #limit phiS label to 3 significant figures
                plt.plot(protein.bibin, ybinNorm, label= f'{protein.name} phiS={convPhiSto_mM(protein.phiS):.3g} mM')
            else:
                plt.plot(protein.bibin, ybinNorm, label= protein.name)

            #plt.plot(protein.bibin, ybinNorm, label= protein.name[11:])


    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$T^*$')
    if phiS==1:
        plt.title(f'{protein_list[0].name[:6]} and Salt Dependence')
    else: plt.title(f'{protein_list[0].name[:6]} and mutants')
    plt.ylim(0.95 * min_ybin, 1.05 * max_ybin)
    plt.xlim(0.0, 1.1 * max_phibin)
    plt.legend(fontsize=9,loc='upper right')
    run_saver(plt,protein_list)
    plt.show()
def run_saver(plot,proteinlist):
    save_bool = input("do you want to save this plot? (Y/N) ").strip().upper()
    if save_bool=='Y':
        today = datetime.today().strftime('%Y-%m-%d')
        if proteinlist[0].rg == 0:
            filename = f"{proteinlist[0].name[:6]}&mutants_fgRPA_w2Pred_{today}.png"
        else:
            if proteinlist[0].crowding_toggle == 1:
                filename = f"{proteinlist[0].name[:6]}&mutants_rgRPA_w2Pred_{today}_crowded_{proteinlist[0].epsC}.png"
            else: filename = f"{proteinlist[0].name[:6]}&mutants_rgRPA_w2Pred_{today}.png"
        savedir = r'C:\Users\Nickl\PycharmProjects\Researchcode (1) (1)\CurrentProjects\PS_ChiFitting_and_ML\using_MLed_w2s\FH_PhaseDiagrams'

        fullpath = os.path.join(savedir,filename)

        plot.savefig(fullpath)
        print(f'plot saved to {fullpath}')



if __name__ == '__main__':
    df = r'C:\Users\Nick\PycharmProjects\Researchcode (1) (1)\CurrentProjects\PS_ChiFitting_and_ML\ML_Lili_w2s\phase_sep_seqs_w2s.csv'
    proteinlist = load_proteins_fromcsv(df,rg=1,phiS=0.0)

    phiSlist1 = [('ddx4n1',convmMtoPhiS(200)),('ddx4n1',convmMtoPhiS(100)),('ddx4n1',convmMtoPhiS(300)),('ddx4n1',convmMtoPhiS(400)),('ddx4n1',convmMtoPhiS(500))]
    phiSlist2 = [('ddx4n1',convmMtoPhiS(200)),('ddx4n1',convmMtoPhiS(100)),('ddx4n1',convmMtoPhiS(300)),('ddx4n1-CS',convmMtoPhiS(100)),('ddx4n1-CS',convmMtoPhiS(300))]

    select_andrun_proteins(proteinlist)
    #select_andrun_proteins_withsalt(proteinlist,phiSlist2)
    #working needs to run

