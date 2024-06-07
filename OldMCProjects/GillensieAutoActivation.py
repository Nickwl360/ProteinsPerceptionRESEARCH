

import numpy as np
import matplotlib.pyplot as plt

# constants
Tmax = 1000000
r = 1e-3
g = 5e-3
gstar = 50e-3
fd = 5e-3
bd = 50
fp = 6e-3
bp = 3e-5
t = 0

# variables
alpha = 1
Na = 0
Na2 = 0
alphastar = 0

initial = [alpha, Na, Na2, alphastar]


def calcProps(nums):
    props = []
    props.append(nums[0] * g)
    props.append(nums[1] * r)
    props.append(nums[1] * (nums[1]-1) * fd)
    props.append(nums[2] * bd)
    props.append(nums[0] * nums[2] * fp)
    props.append(nums[3] * bp)
    props.append(nums[3] * gstar)
    return props

def updatenums(nums, choice):
    numsfix = nums.copy()
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

    return numsfix

def simulation(Tmax, initial):
    t = 0
    clock = 0
    nums = initial

    alphadata = []
    nadata = []
    na2data = []
    astardata = []

    while t < Tmax:

        r1 = np.random.random()
        r2 = np.random.random()
        propensities = calcProps(nums)
        propsum = sum(propensities)
        probs = [x / propsum for x in propensities]
        SUM = 0

        for i in range(len(probs)):
            SUM += probs[i]
            if SUM > r1:
                nums = updatenums(nums,i)
                SUM = -1

        Tau = (1 / propsum) * np.log(1 / r2)
        t += Tau
        clock += Tau
        if clock > 300:
            alphadata.append(nums[0])
            nadata.append(nums[1])
            na2data.append(nums[2])
            astardata.append(nums[3])
            clock = 0
            print(t)

    return alphadata, nadata, na2data, astardata


data = simulation(Tmax, initial)
ts = np.linspace(0, Tmax, len(data[1]))
plt.plot(ts, data[1])
plt.xlabel('time (s)')
plt.ylabel('Number of gene A')
plt.show()
