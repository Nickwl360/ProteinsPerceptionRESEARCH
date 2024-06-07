import numpy as np
from matplotlib import pyplot as plt

e0 = 8.854e-12
C = 1/(4*np.pi*e0)
q = 1.602176634e-19
R = .01
d = .001
U = 1.5
res =12
k = -100
A = np.pi * R * R
Cundergrad = e0 * A /d
QTOT = U * Cundergrad
sig0 = QTOT / A
print(sig0)

sigma_r = np.full((res),sig0 )
rray = np.linspace(R/res,R,res)
#rray=np.delete(rray,0)
print(rray)

iterations =1000
for reps in range(iterations):
    V_r=np.zeros(res)
    for i in range(len(V_r)):
        r0 = rray[i]
        #loops through r, loop through theta
        for j in range(len(rray)):
            r = rray[j]
            theta=2*np.pi/res
            #theta = 0
            while theta < 2*np.pi:
                x = r0 - r*np.cos(theta)
                y = -r*np.sin(theta)
                L = np.sqrt(x*x+y*y)
                L2 = np.sqrt(x*x+y*y+d*d)
                if L != 0:
                    V_r[i]+=sigma_r[i]*(R*r/res)*(2*np.pi/(res))/L
                V_r[i]+=-1*sigma_r[i]*(R*r/res)*(2*np.pi/(res))/L2

                theta += (2*np.pi/(res))


    for i in range(len(sigma_r)):
        #V_r[i]*=C
        sigma_r[i] += k*(V_r[i]-U/2)
    print(sigma_r,'CHARGEEEEEEEEEEEE00')
    print(V_r, ' VVVVVVVVVVVVVVVVVV')

    if np.allclose(V_r, U/2, rtol=(U/2)/100):
        print("Converged after", reps, "iterations.")
        break
sigma_r/= np.max(sigma_r)

sigma_rfinal = sigma_r
print(sigma_rfinal)

def calcQonPlate(sig_r):
    qtot=0
    for i in range(len(rray)):
        r = rray[i]
        qtot += sigma_rfinal[i] * r/R * R/res * 2 * np.pi
    return qtot
def getEfield(sig_r):
    N = 100
    xs = np.linspace(-1.05*R, 1.05*R, N)
    ys = np.linspace(-d,d, N)
    Exs = np.zeros((len(xs),len(ys)))
    Eys = np.zeros((len(xs),len(ys)))
    for i in range(len(xs)):
        for j in range(len(ys)):
            x = xs[i]
            y = ys[j]
            Ex,Ey = calcE(x,y,sig_r)
            Exs[i,j]= Ex
            Eys[i,j]= Ey
            if i == 39 and j ==39:
                print(i,j)
                print(x,y, Ex,Ey)

    return xs,ys, Exs, Eys
def calcE(x,y, sig_r):
    Ex = 0
    Ey = 0
    for k in range(len(rray)):
        r = rray[k]
        r -= R/(res*2)
        #theta = 2*np.pi/res
        theta = 2*np.pi/(res*2)
        while theta <= 2 * np.pi:
            #print(theta)
            xp = x - r * np.cos(theta)
            yt= y - d / 2
            zp = -1* r * np.sin(theta)
            yb = y + d / 2
            R1 =np.sqrt(xp**2 + yt**2 + zp**2)
            L1 = (R1) ** 3
            R2=np.sqrt(xp**2 + yb**2 + zp**2)
            L2 = (R2) ** 3
            if k == 0:
                dA = np.pi*r**2/res
            else: dA = np.pi*(r**2- rray[k-1]**2)/res
            #print(x,y , L1,L2, r, theta)
            ##topplate calc
            if L1!=0:
                Ex+= sig_r[k]*dA*xp/L1
                Ey += sig_r[k] * dA * yt / L1
                if x == 0.00010606060606060605 and y==0.0005555555555555557:
                    print('TOP:   ',Ey, yt, R1,L1,yt/L1, r, theta, xp, zp )

            if L2!=0:
                Ex-= sig_r[k]*dA*xp/L2
                Ey-= sig_r[k]*dA*yb/L2
                if x == 0.00010606060606060605 and y==0.0005555555555555557:
                    print('BOT:   ', Ey, yb, R2,L2, yb/L2, r, theta,xp,zp)

            # if L1 != 0:
            #     Ex += sig_r[k] * (r* R / res) * (2 * np.pi / (res)) * (x1) / L1
            #     Ey += sig_r[k] * (r* R / res) * (2 * np.pi / (res)) * (y1) / L1
            # # bottomplatecalc
            # if L2 != 0:
            #     Ex += -1 * sig_r[k] * (r* R / res) * (2 * np.pi / (res)) * (x2) / L2
            #     Ey += -1 * sig_r[k] * (r* R / res) * (2 * np.pi / (res)) * (y2) / L2

            theta += (2 * np.pi / (res))
    return Ex, Ey
def calcU(sig_r):
    u=0
    for i in range(res):
        u+= U/2* sig_r[i] * rray[i]/R * R/res
    return 2* u * np.pi

print(calcU(sigma_r))

###########################FINISH Q 1 ###################################
qtot = calcQonPlate(sigma_r)/C
Csim = qtot / (U/2)
print(Cundergrad, 'theory cap')
print(Csim, 'sim cap')
#########################PLOTTING PT 1#####################################
plt.plot(rray,sigma_rfinal)
plt.title('Equilibrium Sigma(r) of Circular Plate')
plt.xlabel('r from center (m)')
plt.ylabel('Sigma(r) Density')
plt.show()

#########################PART 2 ##################################
#sigma_r = np.full((res),sig0)

# xs,ys,us,vs, = getEfield(sigma_rfinal)
# plt.streamplot(xs,ys,us,vs,density=.8)
# plt.xlim((-3*R,3*R))
# plt.ylim((-3*d/2,3*d/2))
# plt.show()


