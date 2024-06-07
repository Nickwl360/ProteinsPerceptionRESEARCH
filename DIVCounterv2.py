import numpy as np
import glob as glob
import re
import matplotlib.pyplot as plt
import os

MAX = 89
m = 6
celldatapath = 'celldivdata/'
# data1=np.load(celldatapath+'pbs4.npy')
# print(data1[:,1])

andrewnodivcount = np.load('trajectory_counts_nodiv-3.npy')
andrewdivcount = np.load('trajectory_counts_div-3.npy')
count=0

for path in os.listdir(celldatapath):
    # check if current path is a file
    if os.path.isfile(os.path.join(celldatapath, path)):
        count += 1

class Cell:
    def __init__(self, cell_id,ts, ns):
        self.cell_id = cell_id
        self.alive_times = ts
        self.protein_numbers = ns
    def find_time_index(self, time):
        indices = np.where(self.alive_times == time)[0]
        if len(indices) == 1:
            return int(indices[0])
        else:
            return -1
nodivcounts = np.zeros((MAX + 1, MAX + 1))
divcounts = np.zeros((MAX + 1, MAX + 1, MAX + 1))

cells = []
timeslist=[]
ids=[]
checked=[]


for dt in glob.glob(celldatapath + 'pbs*.npy'):
    a = np.load(dt)
    f_id = int(re.search(r'\d+', dt).group())
    times = a[:, 0]
    nums = a[:, 1]
    divmarker = -1
    cells.append(Cell(f_id,times,nums))
    ids.append(f_id)
L = len(ids)


def countnodiv(cell,start):
    t = cell.alive_times
    next = m - start
    if father !=1:
        t = t[int(next+1):]
    left = 0
    for i in range(len(t)+1):
        if i % m == 0 and (i+m)<len(t):
            indexi = cell.find_time_index(cell.alive_times[i])
            indexj = cell.find_time_index(cell.alive_times[i + m])
            if indexj != -1:
                nodivcounts[int(cell.protein_numbers[indexi])][int(cell.protein_numbers[indexj])] += 1
            else:
                for j in range(0,m +1):
                    test = indexj - j
                    if cell.find_time_index(test) != -1:
                        left = j
                        break

    return  left
def countdiv(d1,d2,start):
    d1c=0
    d2c=0
    fc =0
    father = d1/2
    for cell in cells:
        if cell.cell_id == d1:
            d1c = cell
        if cell.cell_id == d2:
            d2c = cell
        if cell.cell_id == father:
            fc = cell

        if d1c!=0 and d2c!=0 and fc !=0:
            fct = fc.alive_times
            fct = [x for x in fct if x != -1]
            end = m - start
            L = len(fct)-1
            if L>end:
                tf = fct[L - start]
            else: break
            t1 = d1c.alive_times
            t1 = t1[1:]
            if len(t1)>6:
                t1 = t1[end]
                indexk = d1c.find_time_index(t1)

            else: break

            t2 = d2c.alive_times
            t2 = t2[1:]
            if len(t2)>6:
                t2 = t2[end]
                indexl = d2c.find_time_index(t2)
            else: break

            indexi = fc.find_time_index(tf)
            if indexi != -1 and indexk != -1 and indexl != -1:
                divcounts[int(d1c.protein_numbers[indexk])][
                    int(d2c.protein_numbers[indexl])][
                    int(fc.protein_numbers[indexi])] += 1  # KLI
    return
def setfather():
    new = 1
    yes =0
    check2 = checked
    newnew=0
    while yes ==0:
        new+=1
        if new not in check2 and new in ids:
            newnew = new
            yes = 1

        if new > 106890:
            break

    return newnew
starts = np.zeros(106891)
starts[1]=0
def checkagain(father,start):
    for cell in cells:
        if father in checked and father in ids:
            father = setfather()
            #print(father)

        if cell.cell_id == father:
            if cell.cell_id not in checked:
                start = starts[father+1]
                left = countnodiv(cell,start)
                checked.append(cell.cell_id)
                start = left
                starts[father+1]=start
                d1 = father * 2
                d2 = father * 2 + 1
                countdiv(d1,d2,start)
                if d1 not in checked and d1 in ids:
                    father = d1
                    starts[d1+1]=start
                elif d2 not in checked and d2 in ids:
                    father = d2
                    starts[d2+1]=start


    return father, start

father = 1
start = 0
print(L)
while len(checked)<=L-1:
    father, start = checkagain(father,start)
    print(father,start)
    print(len(checked))

print(len(checked))
np.save('divv2.npy', divcounts)
np.save('ndivv2.npy', nodivcounts)

# nodiv = np.transpose(nodivcounts)
# print(andrewnodivcount[1,:])
# print(nodiv[1,:])

# print(andrewdivcount[1, :, :])
# print(divcounts[1, :, :])
nodiv =np.load('ndivv2.npy')
plt.imshow(andrewnodivcount-nodiv)
plt.legend()
plt.show()
