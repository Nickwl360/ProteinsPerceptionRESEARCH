import sys
import numpy as np


file = sys.argv[1]
toprint =np.load(file)
np.set_printoptions(threshold=np.inf)
print(toprint)



