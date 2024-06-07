import numpy as np
import matplotlib.pyplot as plt

# constants
Tmax = 1000000
g = 5e-3
gstar = 50e-3
r=1e-3
dl = 3e-5
dh=1e-5
fd = 5e-3
bd = 50
fp = 6e-3
bp = 3e-5
beta=1
t = 0

class cell:
    def __init__(self, initial):
        self.mu= [initial[0]]
        self.alpha = [initial[1]]
        self.A=[initial[2]]
        self.A2=[initial[3]]
        self.alphastar=[initial[4]]
    def append_nums(self,nextvalues):
        self.mu.append(nextvalues[0])
        self.alpha.append(nextvalues[1])
        self.A.append(nextvalues[2])
        self.A2.append(nextvalues[3])
        self.alphastar.append(nextvalues[4])
    def getlast(self):
        return self.alpha[-1],self.A[-1],self.A2[-1],self.alphastar[-1]
    def get_A_ray(self):
        return np.array(self.A)

##CELL1
first=[1,1,0,0,0]
Cells = [cell(first)]

def calcProps(nums):
    props = []
    dA= dl + (dh-dl)/(1+np.exp(beta*(25-nums[1])))
    props.append(nums[0] * g)
    props.append(nums[1] * r)
    props.append(nums[1] * (nums[1]-1) * fd)
    props.append(nums[2] * bd)
    props.append(nums[0] * nums[2] * fp)
    props.append(nums[3] * bp)
    props.append(nums[3] * gstar)
    props.append(dA)

    return props

def updatenums(nums, choice,Cells,cellnum,divided):
    numsfix = nums.copy()
    new = 0
    if choice == 0:
        numsfix[1] += 1
    elif choice == 1:
        numsfix[1] -= 1
    elif choice == 2:
        numsfix[1] -= 2
        numsfix[2] += 1
    elif choice == 3:
        numsfix[1] += 2
        numsfix[2] -= 1
    elif choice == 4:
        numsfix[0] -= 1
        numsfix[2] -= 1
        numsfix[3] += 1
    elif choice == 5:
        numsfix[0] += 1
        numsfix[2] += 1
        numsfix[3] -= 1
    elif choice == 6:
        numsfix[1] += 1
    elif choice ==7:
        divided.append(Cells[cellnum].mu)
        new = newcells(Cells[cellnum].A)

        newargs1 = [Cells[cellnum].mu*2,1,new[0],0,0]
        newargs2 = [Cells[cellnum].mu*2+1,1,new[1],0,0]

        Cells.append(cell(newargs1))
        Cells.append(cell(newargs2))
    return numsfix,divided
def newcells(NA):
        if NA%2 == 0:  ###REFIT TO BINOMIAL
            A1= NA/2
            A2= NA/2
        elif NA%2==1:
            A1 = np.ceil(NA/2)
            A2 = np.floor(NA/2)
        return A1,A2

def simulation(Tmax, Cells):
    t = 0
    clock = 0
    divided = []

    while t < Tmax:
        for i in range(len(Cells)):
            r1 = np.random.random()
            r2 = np.random.random()
            nums = Cells[i].getlast()
            propensities = calcProps(nums)
            propsum = sum(propensities)
            probs = [x / propsum for x in propensities]
            SUM = 0

            for j in range(len(probs)):
                SUM += probs[i]
                if SUM > r1:
                    for k in range(len(divided)):
                        if (Cells[i].mu != divided[k]):
                            nums,divided = updatenums(nums,j,Cells,i,divided)
                            print(nums)
                            Cells[i].append_nums(nums)
                            SUM = -1
            Tau = (1 / propsum) * np.log(1 / r2)
            t += Tau
            clock += Tau
            if clock > 300:
                #UPDATE
                clock = 0
                #print(t)

    return

simulation(Tmax, Cells)
cell1 = Cells[0].get_A_ray()
ts = np.linspace(0, Tmax, len(cell1))
plt.plot(ts, cell1)
plt.xlabel('time (s)')
plt.ylabel('Number of gene A'+str(cell1))
plt.show()