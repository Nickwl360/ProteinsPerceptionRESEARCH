import numpy as np

jochset16 = np.load('toggletimesjochset15.npy')
#jochfwrseed1=np.load('toggletimesfwrseed1.npy')
#jochfwrseed1=np.load('toggletimesfwrseed1set0.npy')
jochfwrseed1=np.load('toggletimesfwrseed1set15ub.npy')


def calcMean(Dist):
    SUM=0
    for i in range(len(Dist)):
        SUM+= Dist[i]

    return SUM/len(Dist)

def calcSecondMoment(Dist,mean):
    moment_sum=0
    for i in range(len(Dist)):
        moment_sum += (Dist[i] - mean) ** 2
    return moment_sum / len(Dist)


def calcThirdMoment(Dist, mean):
    moment_sum = 0
    for i in range(len(Dist)):
        moment_sum += (Dist[i] - mean) ** 3
    return moment_sum / len(Dist)

def printstats(dist):
    mean = calcMean(dist)
    cv = ((calcSecondMoment(dist,mean))**.5)/mean
    g = calcThirdMoment(dist,mean)/((calcSecondMoment(dist,mean))**1.5)

    print('coef of var:', cv)
    print('skew/cv:', g/cv)
    print('mean:',mean)
    print('Stdev:', calcSecondMoment(dist,mean))
    print()
    print('                                                                          ')
    return

print('JOCHEN ORIGINAL GAMMA STATS')
printstats(jochset16)
print('Jochen infered SEED1 STATS')
printstats(jochfwrseed1)
