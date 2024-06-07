import numpy as np
import matplotlib.pyplot as plt

N = 12
D = 100
e0 = 8.854e-12
C = 1/(4*np.pi*e0)
R = .01
d = .001
V = 1.5

def Efield(q, r0, x, y):
    r = ((x-r0[0])**2 + (y-r0[1])**2)**1.5
    return q * (x - r0[0]) / r, q * (y - r0[1]) / r

####SPACEFOR EFIELDS#########
nx, ny = 129, 129
x = np.linspace(-D*R, D*R, nx)
y = np.linspace(-D*d, D*d, ny)
for i in range(len(y)):
    print(y[i], i)
dy = y[1]-y[0]
dx = x[1]-x[0]
X, Y = np.meshgrid(x, y)

#creating capacitor
nq = 2*N
charges = []
sig_r=[0.06862107, 0.09747887, 0.13157902, 0.17142392, 0.21738329, 0.27069797,
 0.33374594, 0.4104122 , 0.50658533, 0.63046972, 0.79197004 ,1.]  #FROM PT1

for i in range(nq):
    if i<N:
        ###lefTside
        index = np.abs(N-1-i)
    else:
        ##RIghtside
        index = np.abs(N-i)
    # print(i,(i*2*R/(nq-1)-R ))
    #2 platez
    charges.append((-1*sig_r[index], (i*2*R/(nq-1)-R, -d/2)))
    charges.append((sig_r[index], (i*2*R/(nq-1)-R, d/2)))

#calc eeverywhere
Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
for charge in charges:
    ex, ey = Efield(*charge, x=X, y=Y)
    Ex += ex
    Ey += ey

def calcUe(Ex,Ey):
    u=0
    x = np.linspace(-2 * R, 2 * R, nx)

    for i in range(nx):
        xv = x[i]
        if xv>0:
            for j in range(ny):
                u+= np.sqrt(Ex[i,j]**2+Ey[i,j]**2)**2 * dx* xv* dy

    return u * e0 * np.pi#/ (4*np.pi * e0)
def calcF(Ex,Ey):
    u=0
    u2 = 0
    x = np.linspace(-D * R, D * R, nx)

    for i in range(len(x)):
        xv = x[i]
        if xv>0:
            u+= ((Ey[i,64]**2-Ex[i,64]**2)) * dx* xv
            u2 += (V/d)**2 * dx*xv
    return u * e0 * np.pi, u2 * e0 * np.pi

print(calcF(Ex,Ey))
#print(calcUe(Ex,Ey))

plt.streamplot(x,y,Ex,Ey,linewidth=1,density=1.5)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.xlim(-D*R,D*R)
plt.ylim(-D*d,D*d)

plt.show()