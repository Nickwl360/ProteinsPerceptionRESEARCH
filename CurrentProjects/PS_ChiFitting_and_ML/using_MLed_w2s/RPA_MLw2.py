from CurrentProjects.PS_ChiFitting_and_ML.RPA_MODEL_functs import *
import CurrentProjects.PS_ChiFitting_and_ML.ML_Lili_w2s
from CurrentProjects.LLPS_rg_RPA.rgRPA_init import pH_qs
import pandas as pd
import csv
import os

class Protein:
    def __init__(self,name, sequence,v2):
        self.name = name
        self.sequence = sequence
        self.v2 = v2
        self.qc ,self.q_list, self.N = self.calculate_props_fromseq(sequence)

    def calculate_props_fromseq(self, sequence):
        q_list= pH_qs(self.sequence,ph=7)
        N = len(q_list)
        qc = abs(sum(q_list))/N
        return qc, q_list,N

    def __repr__(self):
        return (f"Name({self.name},length={self.N},charge={self.qc},v2={self.v2}")

### this method reads from a csv and creates wanted Protein objects based off of it ###
def load_proteins_fromcsv(file_path):
    df = pd.read_csv(file_path)
    proteinList = [Protein(name=row['Name'],sequence=row['Sequence'],v2=row['w2_preds_LLw302']) for _,row in df.iterrows()]
    return proteinList

### a little silly, but this method takes specific input from user for determining which proteins to use for model. ###
### will create a more general purpose method but this works for now ###
def run_selected_proteins(proteinobj_list):
    for indx, protein in enumerate(proteinobj_list):
        print(f"Index: {indx},Name: {protein.name}")

    index_range=input("enter index range to run on model (eg. 0-3 for first 4 proteins")
    start,stop = map(int,index_range.split('-'))
    for protein in proteinobj_list[start:stop+1]:
        run_model_onProtein(protein)
    return

def run_model_onProtein(protein):
    print(f"Processing {protein.name}:")
    print(f"- Sequence Length: {protein.N}")
    print(f"- Total Charge: {protein.qc}")
    print(f"- Charge List: {protein.q_list}")
    print(f"- v2 Value: {protein.v2}")
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


if __name__ == '__main__':
    df = r'C:\Users\Nick\PycharmProjects\Researchcode (1) (1)\CurrentProjects\PS_ChiFitting_and_ML\ML_Lili_w2s\phase_sep_seqs_w2s.csv'
    proteinlist = load_proteins_fromcsv(df)
    run_selected_proteins(proteinlist)


