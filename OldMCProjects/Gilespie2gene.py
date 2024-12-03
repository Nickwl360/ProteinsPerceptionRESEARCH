import numpy as np
import matplotlib.pyplot as plt

# constants
Tmax = 1000000
r = 1e-3
g = .5
gstar = 2.5e-3
d = .5
p = .02
f = 3.5e-6
b = 2e-5
t = 0

# variables
alpha = 1
lowa = 0
A = 0
alphastar = 0
lowb = 0
B = 0
alphaprime = 0

initial = [alpha, lowa, A, alphastar, lowb, B, alphaprime]


def calcProps(nums):
    props = []
    props.append(nums[0] * g)  # 0
    props.append(nums[1] * d)  # 1
    props.append(nums[1] * p)  # 2
    props.append(nums[2] * r)  # 3
    props.append(nums[0] * nums[2] * f)  # 4
    props.append(nums[3] * b)  # 5
    props.append(nums[3] * g)  # 6
    props.append(nums[3] * gstar)  # 7

    props.append(nums[0] * g)  # 8
    props.append(nums[4] * d)  # 9
    props.append(nums[4] * p)  # 10
    props.append(nums[5] * r)  # 11
    props.append(nums[0] * nums[5] * f)  # 12
    props.append(nums[6] * b)  # 13
    props.append(nums[6] * gstar)  # 14
    props.append(nums[6] * g)  # 15

    return props


def updatenums(nums, choice):
    numsfix = nums.copy()
    if choice == 0:
        numsfix[1] += 1
    elif choice == 1:
        numsfix[1] -= 1
    elif choice == 2:
        numsfix[2] += 1
    elif choice == 3:
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
    elif choice == 7:
        numsfix[4] += 1
    elif choice == 8:
        numsfix[4] += 1
    elif choice == 9:
        numsfix[4] -= 1
    elif choice == 10:
        numsfix[5] += 1
    elif choice == 11:
        numsfix[5] -= 1
    elif choice == 12:
        numsfix[0] -= 1
        numsfix[5] -= 1
        numsfix[6] += 1
    elif choice == 13:
        numsfix[0] += 1
        numsfix[5] += 1
        numsfix[6] -= 1
    elif choice == 14:
        numsfix[1] += 1
    elif choice == 15:
        numsfix[4] += 1

    return numsfix


def simulation(Tmax, initial):
    t = 0
    clock = 0
    nums = initial

    alphadata = []
    lowadata = []
    Adata = []
    astardata = []
    lowbdata=[]
    Bdata=[]
    alphaprimedata=[]

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
                nums = updatenums(nums, i)
                SUM = -1

        Tau = (1 / propsum) * np.log(1 / r2)
        t += Tau
        clock += Tau
        if clock > 300:
            alphadata.append(nums[0])
            lowadata.append(nums[1])
            Adata.append(nums[2])
            astardata.append(nums[3])
            lowbdata.append(nums[4])
            Bdata.append(nums[5])
            alphaprimedata.append(nums[6])
            clock = 0
            print(t)

    return alphadata, lowadata, Adata, astardata, lowbdata,Bdata,alphaprimedata


data = simulation(Tmax, initial)
ts = np.linspace(0, Tmax, len(data[2]))
plt.figure()
plt.plot(ts, data[2], c='red')
plt.plot(ts,data[5],c='blue')

plt.xlabel('time (s)')
plt.ylabel('number')
plt.title('Gillespie gene A (red) and gene B(blue)')
plt.show()
