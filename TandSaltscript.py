from OldProteinProjects.ConfunsTandSalt import *

##a1 = a+, a2 = a-
def Fcon(params, seq,t,cs):
    x, a1, a2 = params
    function = F1(a1, a2, seq,t) + F2(a1, a2, seq,t,cs) + F3(a1, a2, seq,t,cs) + F4(a1, a2, seq,t) + F5(x, a1, a2, seq,t,cs)
    return function

# def minXs(seq,k):
# minf = minimize(Fcon, x0=0.9, args=(seq,k,), method="Nelder-Mead")
# minX = minf.x[0]

# return minX
seq = getseq("SCDtests.xlsx")
seq2 = getseq("xij_test_seqs.xlsx")
seq3 = getseq("p0variants.xlsx")

bounds = [(0, 100), (0, 1), (0, 1)]
def mins(seq,t,cs):
    initial_guess = (.5, .7, .7)
    minf = minimize(Fcon, initial_guess, args=(seq, t,cs,),bounds=bounds, method="Nelder-Mead")
    minX = minf.x
    return minX  #(x,a+,a-)
#BIMODAL:OMI = 1.8


#BIMODAL
# cs =1
# ts = np.linspace(-10, 30, 30)
# a2s = []
# for t in ts:
#     mina2 = mins(seq3[4],t,cs)
#     a2s.append(mina2[2])
#
# plt.plot(ts,a2s)
# plt.xlabel("T (Celsius)")
# plt.ylabel("a-")
# plt.title("a- vs. T Bimodel situation")
# plt.show()


#CHANGING T
# cs =1
# ts = np.linspace(1, 90, 30)
# a11 = []
# a110 = []
# a117 = []
# a125 = []
# a130 = []
# for t in ts:
#     mina11 = mins(seq[0], t, cs)
#     mina110 = mins(seq[9], t, cs)
#     mina117 = mins(seq[16], t, cs)
#     mina125 = mins(seq[24], t, cs)
#     mina130 = mins(seq[29], t, cs)
#
#     a11.append(mina11[1])
#     a110.append(mina110[1])
#     a117.append(mina117[1])
#     a125.append(mina125[1])
#     a130.append(mina130[1])
#
# plt.plot(ts, a11, color = 'blue', label = 'sv1')
# plt.plot(ts, a110, color = 'orange',label = 'sv10')
# plt.plot(ts, a117, color = 'green',label = 'sv17')
# plt.plot(ts, a125, color = 'red',label = 'sv25')
# plt.plot(ts, a130, color = 'purple',label = 'sv30')
# plt.xlabel('T (Celsius)')
# plt.ylabel('a+')
# plt.title('a+ as a function of T')
# plt.legend()
# plt.show()

#CHANGING Cs
# t =20
# css = np.linspace(1, 90, 30)
# a11 = []
# a110 = []
# a117 = []
# a125 = []
# a130 = []
# for cs in css:
#     mina11 = mins(seq[0], t, cs)
#     mina110 = mins(seq[9], t, cs)
#     mina117 = mins(seq[16], t, cs)
#     mina125 = mins(seq[24], t, cs)
#     mina130 = mins(seq[29], t, cs)
#
#     a11.append(mina11[1])
#     a110.append(mina110[1])
#     a117.append(mina117[1])
#     a125.append(mina125[1])
#     a130.append(mina130[1])
#
# plt.plot(css, a11, color = 'blue', label = 'sv1')
# plt.plot(css, a110, color = 'orange',label = 'sv10')
# plt.plot(css, a117, color = 'green',label = 'sv17')
# plt.plot(css, a125, color = 'red',label = 'sv25')
# plt.plot(css, a130, color = 'purple',label = 'sv30')
# plt.xlabel('Concentration of Salt (mM)')
# plt.ylabel('a+')
# plt.title('a+ as a function of Cs')
# plt.legend()
# plt.show()
