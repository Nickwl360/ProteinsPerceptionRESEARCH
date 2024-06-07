import numpy as np
import glob as glob
import re
import matplotlib.pyplot as plt

MAX=89
m=6
celldatapath = 'celldivdata/'
# data1=np.load(celldatapath+'pbs4.npy')
# print(data1[:,1])

andrewnodivcount = np.load('trajectory_counts_nodiv-3.npy')
andrewdivcount = np.load('trajectory_counts_div-3.npy')

# class Cell:
#     def __init__(self, cell_id,ts,ns,idiv):
#         self.cell_id = cell_id
#         self.alive_times = ts  # Array to store times alive
#         self.protein_numbers = ns
#         self.divided = idiv
#
#     def find_time_index(self, time):
#         indices = np.where(self.alive_times == time)[0]
#         if len(indices) == 1:
#             return indices[0]
#         else:
#             return -1
#
# checktimes = np.arange(0,669,m)
# nodivcounts = np.zeros((MAX+1,MAX+1))
# divcounts = np.zeros((MAX+1,MAX+1,MAX+1))
# cells=[]
#
# for dt in glob.glob(celldatapath+'pbs*.npy'):
#     a=np.load(dt)
#     f_id = int(re.search(r'\d+', dt).group())
#     times = a[:,0]
#     nums = a[:,1]
#     divmarker = -1
#
#
#
#
#     if times[-1] == divmarker:
#         lastdix = np.where(times==divmarker)[0][0]-1
#         idiv = 1
#     elif times[-1] != divmarker:
#         lastdix = len(times)-1
#         idiv = 0
#
#
#     divt = times[lastdix]
#     ts = times[:lastdix]
#     ns = nums[:lastdix]
#     ts = ts//300
#     if len(ts)!=0:
#         cells.append(Cell(f_id,ts,ns,idiv))
#
#
#
# #########COUNTING
# for cell in cells:
#     if cell.divided ==0 or cell.divided==1:
#         for t in cell.alive_times:
#             if t%m==0:
#                 indexi = cell.find_time_index(t)
#                 indexj = cell.find_time_index(t+m)
#                 if indexj != -1:
#                     print(cell.cell_id, cell.alive_times[indexi], cell.alive_times[indexj])
#                     nodivcounts[int(cell.protein_numbers[indexi])][int(cell.protein_numbers[indexj])]+=1
#
#     if cell.divided ==1:
#         for l in cell.alive_times:
#             for daughter1 in cells:
#                 for daughter2 in cells:
#                     if daughter1.cell_id == cell.cell_id * 2 and daughter2.cell_id == (cell.cell_id*2 + 1):
#                         ###FIX T
#                         if l%m ==0:
#                             indexi = cell.find_time_index(l)
#                             indexk = daughter1.find_time_index(l+m)
#                             indexl= daughter2.find_time_index(l+m)
#                             if indexi !=-1 and indexk !=-1 and indexl !=-1:
#                                 divcounts[int(daughter2.protein_numbers[indexk])][int(daughter2.protein_numbers[indexl])][int(cell.protein_numbers[indexi])] +=1#KLI

# nodiv = np.transpose(nodivcounts)
# print(andrewnodivcount[1,:])
# print(nodiv[1,:])
#
# np.save('nodivv1',nodiv)
# np.save('divv1',divcounts)

nd = np.load('nodivv1.npy')
plt.imshow(andrewnodivcount-nd)
plt.legend()
plt.show()
