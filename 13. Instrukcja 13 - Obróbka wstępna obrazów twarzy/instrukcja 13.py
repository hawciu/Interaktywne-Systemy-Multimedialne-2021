import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

#Zadanie 1 progowanie kompletnie źle zrobione - wyodrebniona miala byc twarz a u pana jest wyodrębnione wszystko...
# miał pana analizowac fragment z twarzą na podstawie kilku plikow


name = "Keanu_Reeves_00"
#YCrCb = cv2.cvtColor(cv2.imread(name + '06' + '.jpg'), cv2.COLOR_BGR2YCrCb)
YCrCb = cv2.cvtColor(cv2.imread('George_W_Bush_0050.jpg'), cv2.COLOR_BGR2YCrCb)

YCrCb = YCrCb[33:202, 66:185]

plt.figure()
histogram1 = cv2.calcHist([YCrCb[:, :, 1]], [0], None, [255], [0, 255])
plt.plot(histogram1)
plt.title('histogramy')
histogram2 = cv2.calcHist([YCrCb[:, :, 2]], [0], None, [255], [0, 255])
plt.plot(histogram2)
plt.legend(['Cr', 'Cb'])
#plt.show()

pikiCb = scipy.signal.find_peaks(histogram2.ravel())[0]
print(pikiCb[0])
pikiCr = scipy.signal.find_peaks(histogram1.ravel())[0]



#minCb = int(min(range(len(histogram2)), key=histogram2.__getitem__) * 1.05)
#maxCb = int(max(range(len(histogram2)), key=histogram2.__getitem__) * 0.95)
#minCr = int(max(range(len(histogram1)), key=histogram1.__getitem__) * 1.05)
#maxCr = int(max(range(len(histogram1)), key=histogram1.__getitem__) * 0.95)
minCb = min(pikiCb) - 10
maxCb = max(pikiCb) + 10
minCr = min(pikiCr) + 10
maxCr = max(pikiCr) + 10
print(min(range(len(histogram2)), key=histogram2.__getitem__), 'minCb:', minCb)
print(max(range(len(histogram2)), key=histogram2.__getitem__), 'maxCb:', maxCb)
print(min(range(len(histogram1)), key=histogram1.__getitem__), 'minCr:', minCr)
print(max(range(len(histogram1)), key=histogram1.__getitem__), 'maxCr:', maxCr)

# minCb = 85
# maxCb = 135
# minCr = 135
# maxCr = 180
print(minCb, maxCb, minCr, maxCr)
result = np.zeros([YCrCb.shape[0], YCrCb.shape[1]])
for i in range(YCrCb.shape[0]):
    for j in range(YCrCb.shape[1]):
        # if (199 < (imgin[i, j, 2] + 0.6 * imgin[i, j, 1]) < 215) and (138 < imgin[i, j, 1] < 178):
        # if 85 < imgin[i, j, 2] < 135 and 135 < imgin[i, j, 1] < 180:
        # Y Cr Cb
        if minCb < YCrCb[i, j, 2] < maxCb and minCr < YCrCb[i, j, 1] < maxCr:
            result[i, j] = 1

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2RGB))
plt.subplot(1, 2, 2)
plt.imshow(result, cmap='Greys_r', interpolation='nearest')
plt.show()




