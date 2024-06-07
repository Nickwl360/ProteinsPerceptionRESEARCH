from TandSaltscript import *
import numpy as np

seq = getseq("p0variants.xlsx")
t = 1.435
cs = 1
a1 = .9610
res = 80
a2s = np.linspace(.64, .78, res)
xs = np.linspace(.4, 1.55, res)
beta = 1.380649e-23 * (t + 273.15)

def calcEs(seq, t, cs):
    energies = np.zeros((len(a2s), len(xs)))
    for i in range(res):
        for j in range(res):
            energies[i][j] = Fcon((xs[j], a1, a2s[i]), seq, t, cs)/beta/len(seq)

    return energies

energies = calcEs(seq[4], t, cs)
min = np.min(energies)
minimum_indices = np.unravel_index(np.argmin(energies), energies.shape)
min_a2 = a2s[minimum_indices[0]]
min_xs = xs[minimum_indices[1]]
print(min, min_a2, min_xs)

# Use np.meshgrid to create cell edges for pcolor
a2s_edges, xs_edges = np.meshgrid(a2s, xs, indexing='ij')
plt.pcolor(a2s_edges, xs_edges, energies, cmap='rainbow_r', vmin=min, vmax=min + 0.0003)
plt.xlabel('a2s')
plt.ylabel('xs')
plt.title("Free energies")
plt.colorbar()  # Add a color bar
plt.show()
#pcolor#