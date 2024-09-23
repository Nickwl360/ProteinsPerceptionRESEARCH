from matplotlib import pyplot as plt
import matplotlib.image as mpimg

#change to desired ALWAYS INCLUDE 'r' IN FRONT
file_path = r'C:\Users\Nickl\PycharmProjects\Researchcode (1) (1)\CurrentProjects\PS_ChiFitting\FH_PhaseDiagrams\AlexA1C_fromw2Pred.png'

img = mpimg.imread(file_path)

plt.imshow(img)
plt.axis('on')
plt.show()