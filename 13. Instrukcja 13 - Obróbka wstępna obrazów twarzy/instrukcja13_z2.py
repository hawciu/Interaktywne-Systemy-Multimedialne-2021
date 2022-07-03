import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_squared_error


suma = np.zeros([112, 92])
for i in range(35, 36):
    for j in range(1, 11):
        suma += plt.imread('s' + str(i) + '/' + str(j) + '.pgm')
suma /= 40
#plt.imsave('suma.png', suma, cmap='gray')

plt.imshow(suma, cmap='gray')
plt.show()
test = cv2.imread('test3.jpg')
testG = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
#plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
#plt.show()

wynik = np.zeros((testG.shape[0]-suma.shape[0])*(testG.shape[1]-suma.shape[1]))
print('suma shape', suma.shape, 'test shape', testG.shape)
print('len', len(wynik))
print('ll', (testG.shape[0]-suma.shape[0]), (testG.shape[1]-suma.shape[1]))
licznik = 0
for i in range(testG.shape[0]-suma.shape[0]):
    for j in range(testG.shape[1]-suma.shape[1]):
        #print(i, j, licznik)
        wynik[licznik] = mean_squared_error(testG[i:i+suma.shape[0], j:j+suma.shape[1]], suma)
        licznik = licznik + 1
xd = min(range(len(wynik)), key=wynik.__getitem__)
print('xd', xd)
print(len(wynik))
print(wynik[len(wynik)-5:len(wynik)-1])


fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
x = int(xd/(testG.shape[1]-suma.shape[1]))
y = (xd % (testG.shape[1]-suma.shape[1]))
print(x, y)
rect = patches.Rectangle((y, x), suma.shape[1], suma.shape[0], linewidth=1, edgecolor='r', facecolor='none')

ax.add_patch(rect)

plt.show()






