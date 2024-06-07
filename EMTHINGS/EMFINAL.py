import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

WIDTH, HEIGHT, DPI = 700, 700, 100
N = 10
e0 = 8.854e-12
C = 1/(4*np.pi*e0)
R = .01
d = .001

def E(q, r0, x, y, z):
    den = ((x-r0[0])**2 + (y-r0[1])**2 + (z-r0[2]))**1.5
    if den !=0:
        return q*C * (x - r0[0]) / den, q*C * (y - r0[1]) / den, q*C * (z - r0[2]) / den
    else: return 0
# Grid of x, y points
nx, ny = 128, 128
x = np.linspace(-2*R, 2*R, nx)
y = np.linspace(-d, d, ny)
z = np.linspace(-2*R,2*R,nx)
X, Y = np.meshgrid(x, y)

#creating capacitor
nq, d = 2*N, .001
charges = []
sig_r=[ 255.68660148,  399.32387982,  575.65766632,  787.69378038, 1042.8096571,
 1356.67637447, 1754.66753772, 2273.12319468, 2956.58024212, 3844.57097813]

for i in range(nq):
    if i<10:
        index = np.abs(N-1-i)
    else:
        index = np.abs(N-i)
    print(i,(i*2*R/(nq-1)-R ))
    charges.append((-1*sig_r[index], (i*2*R/(nq-1)-R, -d/2, i*2*R/(nq-1)-R)))
    charges.append((sig_r[index], (i*2*R/(nq-1)-R, d/2, i * 2*R/(nq-1)-R)))

#calc eeverywhere
Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
for charge in charges:
    ex, ey,ez = E(*charge, x=X, y=Y,z=0)
    Ex += ex
    Ey += ey

def calcUe(Ex,Ey):
    u=0
    for i in range(nx):
        for j in range(ny):
            u+= np.sqrt(Ex[i,j]**2+Ey[i,j]**2)*R/nx * R/ny

    return u * e0/2

print(calcUe(Ex,Ey))

fig = plt.figure(figsize=(WIDTH/DPI, HEIGHT/DPI), )
ax = fig.add_subplot()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Plot the streamlines with an appropriate colormap and arrow style
ax.streamplot(x, y, Ex, Ey,  linewidth=1,
              density=1.5, arrowstyle='->')


ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim(-2*R,2*R)
ax.set_ylim(-d,d)

plt.show()