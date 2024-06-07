import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
from scipy.special import kn

def sinN(n,x):
    return np.sin(n*np.pi*x)
def en(n,x):
    return np.exp(-1*n*np.pi*x)


# Define the x-values for the plot
x = np.linspace(0, 5, 1000)
# Compute the Bessel functions using list comprehension
y = [en(n, x)*sinN(n,x) for n in range(4)]

# Plot the Bessel functions
for i, yi in enumerate(y):
    plt.plot(x, yi, label='e{}(x)'.format(i))
plt.xlabel('x')
plt.ylabel('K(x)')
plt.title('Exponential * Sinusoidal Functions')
# plt.ylim(0,10)
# plt.xlim(0,20)
plt.legend()
plt.show()