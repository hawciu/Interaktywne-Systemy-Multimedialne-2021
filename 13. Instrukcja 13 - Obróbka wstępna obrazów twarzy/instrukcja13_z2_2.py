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

#plt.imshow(suma, cmap='gray')
#plt.show()
test = cv2.imread('test3.jpg')
testG = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
#plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
#plt.show()
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
methodsstr = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in range(len(methods)):
    img = testG.copy()
    # Apply template Matching
    #method = eval(m)
    res = cv2.matchTemplate(testG.astype(np.float32), suma.astype(np.float32), methods[m])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

    top_left = min_loc
    top_left = max_loc
    bottom_right = (top_left[0] + suma.shape[1], top_left[1] + suma.shape[0])
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(methodsstr[m])
    plt.show()





