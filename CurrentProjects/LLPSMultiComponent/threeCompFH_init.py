import numpy as np

################CONSTANTS######################
#chi11= .66
#chi22= .66

#panel1
#chi11,chi22,chi12 = 1.3,.5,.8
#panel2
#chi11,chi22,chi12 = 1.3,.5,1
#panel3
#chi11,chi22,chi12 = 1.3,.5,1.3
#panel4
chi11,chi22,chi12 = 1.3,.9,1.3
#panel5
#chi11,chi22,chi12 = 1.3,.9,.9

#chi12 = np.array([.72, .645, .63])
#chi12 = .63
#chi12= np.sqrt(chi11)*np.sqrt(chi22)

##########LENGTHS########
#N1, N2 = 50,50
N1,N2 = 10,10


def initialize():
    print("3 COMPONENT FH MODEL: \nchi11: ", chi11, "\nchi22: ", chi22, "\nchi12: ", chi12, "\nN1, N2: ", N1, N2 )

    return
