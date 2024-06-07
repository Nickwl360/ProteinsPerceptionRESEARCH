import numpy as np
from matplotlib import pyplot as plt

e0 = 8.854e-12
q = 1.602176634e-19
R = 0.01
d = 0.001
U = 1.5
res =10
final = res-1

rlist = np.linspace(R/(res-1), R, res-1)
print(rlist)

thetalist = np.linspace(0,2*np.pi, res)
thetalist = np.delete(thetalist,final) ####ALL THETA EXCEPT 2PI
print(thetalist)

Qtot = U * e0 * np.pi * R * R / d
charges = Qtot / q
spaces = (res - 1)*(res-1 )-1
qperspaceinitial =  charges / spaces

q_density = np.full((res-1, res-1), qperspaceinitial)  ########## r1 -> R,  0 -> thetafinal
r0qdensity = qperspaceinitial
iteration_limit = 650  # Set a reasonable iteration limit
convergence_limit = 0.01  # Set the convergence limit, e.g., 1%

for iteration in range(iteration_limit):
    phi = np.zeros((res - 1, res - 1))
    phir0 = 0
    print(iteration)
    ###CALC PHIR0 from rest
    for i in range(res-1):   #rs
        r = rlist[i]
        phir0 += ((1*(res-1))/(4*np.pi*e0)) *q* q_density[i,0] / r

    ###CALC PHI[i,j]
    for i in range(res-1):
        r= rlist[i]
        phi[i,:] += (1/(4*np.pi*e0)) *q* r0qdensity / r

        ##position of others
        for k in range(res-1):
            for l in range(res-1):
                r2 = rlist[k]
                theta = thetalist[l]
                x = r-r2*np.cos(theta)
                y = r2*np.sin(theta)
                L = np.sqrt(x*x+y*y)
                if L != 0:
                    phi[i,:] += (1/(4*np.pi*e0)) * q* q_density[k,0] / L
    dq = 100
    m =2.5
    if phir0 > U:
        r0qdensity -= dq * (phir0/U)**m
        if r0qdensity < 0:
            r0qdensity = 0
    elif phir0 < U:
        r0qdensity += dq * np.abs(U/phir0)**m

    for i in range(res-1):
        if phi[i, 0] > U:
            q_density[i, :] -= dq * (phi[i, 0]/U)**m

            if q_density[i,0]<0:
                q_density[i,:]=0
        elif phi[i, 0] < U:
            q_density[i, :] += dq * np.abs(U/phi[i, 0])**m



    # total_charge = (np.sum(q_density) + r0qdensity)
    # q_density *= (charges / total_charge)
    # r0qdensity *= charges / total_charge
    # total_charge = (np.sum(q_density) + r0qdensity)
    # print(total_charge/charges, 'EQUALS 1???????')


    # Check for convergence
    if np.allclose(phi, U, rtol=convergence_limit ):
        print("Converged after", iteration, "iterations.")
        break

Q_total = np.sum(q_density) * q + r0qdensity*q
C_numerical = Q_total / U

# Calculate capacitance using the formula C = epsilon * A / d
A = np.pi * R * R
C_formula = e0 * A / d

# Print results
print('phi = ', phir0, ' at r = 0, q# = ', r0qdensity)
for i in range(res-1):
        print('phi = ',phi[i,0],' at r = ',rlist[i], 'q# =', q_density[i,0])
print("Numerical Capacitance:", C_numerical)
print("Formula Capacitance:", C_formula)

chargedist = [r0qdensity]
rs=[0]
for r in range(res-1):
    chargedist.append(q_density[r,0])
    rs.append(rlist[r])
plt.plot(rs,chargedist)
plt.title('Q#(r) at Phi = 1.5V (almost) everywhere')
plt.xlabel('r from center (m)')
plt.ylabel('# of charges')
plt.show()
