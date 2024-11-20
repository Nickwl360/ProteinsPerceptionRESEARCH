from OldMCProjects.OldPerceptionRelated.MCBrainTime import *
import numpy as np
#import matplotlib.pyplot as plt

###firsttry
# initial=(0,24)
# epsilon=1
# (halpha, ha, ka, kb) = (.2, .01, .15, .15)
# params = (halpha, ha, halpha - epsilon, ha + epsilon, ka, kb) ,3000

##take2
# initial=(0,24)
# epsilon=1
# (halpha, ha, ka, kb) = (.25, .025, .13, .15) e = .7, max=3000

#take3
initial=(0,24)
epsilon=1.1
(halpha, ha, ka, kb) = (.0, .0, .15, .15)
params = (halpha, ha, halpha - epsilon, ha + epsilon, ka, kb)

Tmax = 6000
samples = 10000

def sampletimes(initial,params,samples):
    fliptime=0
    i = 0
    toggletimes=[]
    while i < samples:
        a,b = simulation(initial,params,Tmax)
        check=0
        for t in range(len(a)):
            if a[t]>b[t]:
                check+=1
                if check ==5:
                    fliptime=t-5
                    
                    break

        i+=1
        toggletimes.append(fliptime)
    return toggletimes

nbins=600
toggletimes =sampletimes(initial,params,samples)
np.save('MCBrainflips.npy',toggletimes)
#plt.hist(toggletimes,bins=nbins)
#plt.show()
