#Próbkowanie i zmiana rozmiaru obrazu, kwantyzacja oraz dithering
import scipy.interpolate
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image

palette1 = np.array(np.linspace(0, 1, 2**1))
palette2 = np.array(np.linspace(0, 1, 2**2))
palette4 = np.array(np.linspace(0, 1, 2**4))
M1 = np.array([[0, 2], [3, 1]])
M2 = np.vstack((np.hstack((M1 * 4, M1 * 4 + 2)), np.hstack((M1 * 4 + 3, M1 * 4 + 1))))
M3 = np.vstack((np.hstack((M2 * 4, M2 * 4 + 2)), np.hstack((M2 * 4 + 3, M2 * 4 + 1))))
M1 = M1 / (2**1)**2 - 0.5
M2 = M2 / (2**2)**2 - 0.5
M3 = M3 / (2**4)**2 - 0.5  # po co to  -0,5 do <-0.5,0.5>?


def metodanajblizszegosasiada(imgin, scalein):
    sizex = len(imgin)
    sizey = len(imgin[0])
    scale = scalein/100
    targetsizex = int(sizex * scale)
    targetsizey = int(sizey * scale)
    result = np.zeros((targetsizex, targetsizey, 3), float)
    i = 0
    for x in range(targetsizex):
        for y in range(targetsizey):
            q12 = np.array([int(x / scale), int(y / scale)])
            q22 = np.array([int(x / scale) + 1, int(y / scale)])
            q11 = np.array([int(x / scale), int(y / scale) + 1])
            q21 = np.array([int(x / scale) + 1, int(y / scale) + 1])
            if x != 0 and y != 0 and x != targetsizex-1 and y != targetsizey-1:
                point = np.array([x, y])
                points = np.array([q12, q22, q11, q21])
                minimum = np.argmin(np.array([np.linalg.norm(q12 - point),
                                    np.linalg.norm(q22 - point),
                                    np.linalg.norm(q11 - point),
                                    np.linalg.norm(q21 - point)]))
                result[x][y][0] = imgin[points[minimum][0]][points[minimum][1]][0]  # R
                result[x][y][1] = imgin[points[minimum][0]][points[minimum][1]][1]  # G
                result[x][y][2] = imgin[points[minimum][0]][points[minimum][1]][2]  # B
            # gorna krawedz
            elif x == 0 and y != 0 and y != targetsizey-1:
                point = np.array([x, y])
                points = np.array([q12, q22])
                minimum = np.argmin(np.array([np.linalg.norm(q12 - point),
                                    np.linalg.norm(q22 - point)]))
                result[x][y][0] = imgin[points[minimum][0]][points[minimum][1]][0]  # R
                result[x][y][1] = imgin[points[minimum][0]][points[minimum][1]][1]  # G
                result[x][y][2] = imgin[points[minimum][0]][points[minimum][1]][2]  # B
            # dolna krawedz
            elif x == targetsizex-1 and y != 0 and y != targetsizey-1:
                point = np.array([x, y])
                points = np.array([q21, q11])
                minimum = np.argmin(np.array([np.linalg.norm(q11 - point),
                                    np.linalg.norm(q21 - point)]))
                result[x][y][0] = imgin[points[minimum][0]][points[minimum][1]][0]  # R
                result[x][y][1] = imgin[points[minimum][0]][points[minimum][1]][1]  # G
                result[x][y][2] = imgin[points[minimum][0]][points[minimum][1]][2]  # B
            # lewa krawedz
            elif y == 0 and x != 0 and x != targetsizex-1:
                point = np.array([x, y])
                points = np.array([q11, q21])
                minimum = np.argmin(np.array([np.linalg.norm(q11 - point),
                                    np.linalg.norm(q21 - point)]))
                result[x][y][0] = imgin[points[minimum][0]][points[minimum][1]][0]  # R
                result[x][y][1] = imgin[points[minimum][0]][points[minimum][1]][1]  # G
                result[x][y][2] = imgin[points[minimum][0]][points[minimum][1]][2]  # B
            # prawa krawedz
            elif y == targetsizey-1 and x != 0 and x != targetsizex-1:
                point = np.array([x, y])
                points = np.array([q12, q22])
                minimum = np.argmin(np.array([np.linalg.norm(q12 - point),
                                    np.linalg.norm(q22 - point)]))
                result[x][y][0] = imgin[points[minimum][0]][points[minimum][1]][0]  # R
                result[x][y][1] = imgin[points[minimum][0]][points[minimum][1]][1]  # G
                result[x][y][2] = imgin[points[minimum][0]][points[minimum][1]][2]  # B
    #narozniki - srednie dwoch sasiadow
    result[0][targetsizey-1][0] = (result[1][targetsizey-1][0]+result[0][targetsizey-2][0])/2
    result[0][targetsizey-1][1] = (result[1][targetsizey-1][0]+result[0][targetsizey-2][1])/2
    result[0][targetsizey-1][2] = (result[1][targetsizey-1][0]+result[0][targetsizey-2][2])/2

    result[targetsizex-1][targetsizey-1][0] = (result[targetsizex-2][targetsizey-1][0]+result[targetsizex-1][targetsizey-2][0])/2
    result[targetsizex-1][targetsizey-1][1] = (result[targetsizex-2][targetsizey-1][1]+result[targetsizex-1][targetsizey-2][1])/2
    result[targetsizex-1][targetsizey-1][2] = (result[targetsizex-2][targetsizey-1][2]+result[targetsizex-1][targetsizey-2][2])/2

    result[targetsizey-1][0][0] = (result[targetsizey-2][0][0]+result[targetsizey-1][1][0])/2
    result[targetsizey-1][0][1] = (result[targetsizey-2][0][0]+result[targetsizey-1][1][1])/2
    result[targetsizey-1][0][2] = (result[targetsizey-2][0][0]+result[targetsizey-1][1][2])/2

    result[0][0][0] = (result[1][0][0]+result[0][1][0])/2
    result[0][0][1] = (result[1][0][0]+result[0][1][1])/2
    result[0][0][2] = (result[1][0][0]+result[0][1][2])/2
    return result


def metodainterpolacjidwuliniowej(imgin, scalein):
    basex = np.linspace(0, 1, int(len(imgin)))
    basey = np.linspace(0, 1, int(len(imgin[0])))
    z = imgin[:, :, 0]
    xyz = scipy.interpolate.RectBivariateSpline(basex, basey, z)
    scale = scalein/100
    targetx = np.linspace(0, 1, int(len(imgin)*scale))
    targety = np.linspace(0, 1, int(len(imgin[0])*scale))
    result = np.zeros(((int(len(imgin) * scale)), int((len(imgin[0]) * scale)), 3))
    for i in range(len(targetx)):
        for j in range(len(targety)):
            tmp = xyz.__call__(targetx[i], targety[j])

            # jesli gamon wyinterpoluje za float
            if tmp > 1:
                tmp = 1.0
            if tmp < 0:
                tmp = 0

            result[i][j][0] = tmp
            result[i][j][1] = result[i][j][0]
            result[i][j][2] = result[i][j][0]

    return result


def metodasredniej(imgin, scalein):
    sizex = len(imgin)
    print(sizex)
    sizey = len(imgin[0])
    scale = scalein/100
    print(scale)
    targetsizex = int(sizex * scale)
    targetsizey = int(sizey * scale)
    result = np.zeros((targetsizex, targetsizey, 3), float)
    for x in range(targetsizex):
        for y in range(targetsizey):
            # wspolrzedne w oryginlnym obrazie
            xy = np.array([int(x / scale), int(y / scale)])
            if x != 0 and y != 0 and x != targetsizex-1 and y != targetsizey-1:
                values = np.array([
                    imgin[xy[0] - 1][xy[1] - 1][0], imgin[xy[0] - 1][xy[1]][0], imgin[xy[0] - 1][xy[1] + 1][0],
                    imgin[xy[0]][xy[1] - 1][0], imgin[xy[0]][xy[1]][0], imgin[xy[0]][xy[1] + 1][0],
                    imgin[xy[0] + 1][xy[1] - 1][0], imgin[xy[0] + 1][xy[1]][0], imgin[xy[0] + 1][xy[1] + 1][0],
                ])
            # gorna krawedz
            elif x == 0 and y != 0 and y != targetsizey-1:
                values = np.array([
                    imgin[xy[0]][xy[1] - 1][0], imgin[xy[0]][xy[1]][0], imgin[xy[0]][xy[1] + 1][0],
                    imgin[xy[0] + 1][xy[1] - 1][0], imgin[xy[0] + 1][xy[1]][0], imgin[xy[0] + 1][xy[1] + 1][0],
                ])
            # dolna krawedz
            elif x == targetsizex-1 and y != 0 and y != targetsizey-1:
                values = np.array([
                    imgin[xy[0] - 1][xy[1] - 1][0], imgin[xy[0] - 1][xy[1]][0], imgin[xy[0] - 1][xy[1] + 1][0],
                    imgin[xy[0]][xy[1] - 1][0], imgin[xy[0]][xy[1]][0], imgin[xy[0]][xy[1] + 1][0],
                ])
            # lewa krawedz
            elif y == 0 and x != 0 and x != targetsizex-1:
                values = np.array([
                    imgin[xy[0] - 1][xy[1]][0], imgin[xy[0] - 1][xy[1] + 1][0],
                    imgin[xy[0]][xy[1]][0], imgin[xy[0]][xy[1] + 1][0],
                    imgin[xy[0] + 1][xy[1]][0], imgin[xy[0] + 1][xy[1] + 1][0],
                ])
            # prawa krawedz
            elif y == targetsizey-1 and x != 0 and x != targetsizex-1:
                values = np.array([
                    imgin[xy[0] - 1][xy[1] - 1][0], imgin[xy[0] - 1][xy[1]][0],
                    imgin[xy[0]][xy[1] - 1][0], imgin[xy[0]][xy[1]][0],
                    imgin[xy[0] + 1][xy[1] - 1][0], imgin[xy[0] + 1][xy[1]][0],
                ])
            # narozniki
            elif x == 0 and y == 0:
                values = np.array([
                    imgin[xy[0]][xy[1]][0], imgin[xy[0]][xy[1] + 1][0],
                    imgin[xy[0] + 1][xy[1]][0], imgin[xy[0] + 1][xy[1] + 1][0],
                ])
            elif x == 0 and y == targetsizey-1:
                values = np.array([
                    imgin[xy[0]][xy[1] - 1][0], imgin[xy[0]][xy[1]][0],
                    imgin[xy[0] + 1][xy[1] - 1][0], imgin[xy[0] + 1][xy[1]][0],
                ])
            elif x == targetsizex-1 and y == 0:
                values = np.array([
                    imgin[xy[0] - 1][xy[1]][0], imgin[xy[0] - 1][xy[1] + 1][0],
                    imgin[xy[0]][xy[1]][0], imgin[xy[0]][xy[1] + 1][0],
                ])
            elif x == targetsizex-1 and y == targetsizey-1:
                values = np.array([
                    imgin[xy[0] - 1][xy[1] - 1][0], imgin[xy[0] - 1][xy[1]][0],
                    imgin[xy[0]][xy[1] - 1][0], imgin[xy[0]][xy[1]][0],
                ])
            result[x][y][0] = np.mean(values)
            result[x][y][1] = result[x][y][0]
            result[x][y][2] = result[x][y][0]
    return result


def colorfit(valuein, palette):
    value = palette[np.argmin(np.absolute(palette - valuein))]
    return value


def ditheringzorganizowany(imgin, bity):
    print ("dithering start ", bity)
    if bity == 1:
        palette = palette1
    if bity == 2:
        palette = palette2
    if bity == 4:
        palette = palette4
    sizex = len(imgin)
    sizey = len(imgin[0])
    result = np.zeros((sizex, sizey, 3), float)
    for x in range(sizex):
        for y in range(sizey):
            xm = np.mod(x, len(M2))
            ym = np.mod(y, len(M2))
            result[x][y][0] = colorfit(imgin[x][y][0] + M2[xm][ym], palette)
            result[x][y][1] = result[x][y][0]
            result[x][y][2] = result[x][y][0]

    return result


def ditheringfloydsteinberg(imgin, palettein):
    sizex = len(imgin)
    sizey = len(imgin[0])
    result = np.copy(imgin)
    for x in range(sizex):
        for y in range(sizey):
            oldpixel = np.copy(result[x][y][0])
            newpixel = colorfit(oldpixel, palettein)
            result[x][y][0] = newpixel
            quant_error = oldpixel - newpixel
            if x != sizex-1 and y != sizey-1:
                tmp = result[x + 1][y][0] + quant_error * 7/16
                result[x + 1][y][0] = tmp
                result[x + 1][y][1] = tmp
                result[x + 1][y][2] = tmp

                tmp = result[x - 1][y + 1][0] + quant_error * 3/16
                result[x - 1][y + 1][0] = tmp
                result[x - 1][y + 1][1] = tmp
                result[x - 1][y + 1][2] = tmp

                tmp = result[x][y + 1][0] + quant_error * 5/16
                result[x][y + 1][0] = tmp
                result[x][y + 1][1] = tmp
                result[x][y + 1][2] = tmp

                tmp = result[x + 1][y + 1][0] + quant_error * 1/16
                result[x + 1][y + 1][0] = tmp
                result[x + 1][y + 1][1] = tmp
                result[x + 1][y + 1][2] = tmp

            #krawedz dolna
            elif x == sizex-1 and y != sizey-1:
                tmp = result[x - 1][y + 1][0] + quant_error * 3/16
                result[x - 1][y + 1][0] = tmp
                result[x - 1][y + 1][1] = tmp
                result[x - 1][y + 1][2] = tmp

                tmp = result[x][y + 1][0] + quant_error * 5/16
                result[x][y + 1][0] = tmp
                result[x][y + 1][1] = tmp
                result[x][y + 1][2] = tmp

            #krwedz prawa
            elif x != sizex-1 and y == sizey-1:
                tmp = result[x + 1][y][0] + quant_error * 7/16
                result[x + 1][y][0] = tmp
                result[x + 1][y][1] = tmp
                result[x + 1][y][2] = tmp
    result = np.clip(result, 0 , 1)
    return result


img = plt.imread('0009.png')
#img = img[1600:1900, 850:1050]
hehe = ditheringfloydsteinberg(img, palette1)
hehe2 = ditheringfloydsteinberg(img, palette2)
hehe3 = ditheringfloydsteinberg(img, palette4)
plt.subplot(1,3,1)
plt.title('dithering floyd-steinberg 1 bit')
plt.imshow(hehe)
plt.subplot(1,3,2)
plt.title('dithering floyd-steinberg 2 bity')
plt.imshow(hehe2)
plt.subplot(1,3,3)
plt.title('dithering floyd-steinberg 4 bity')
plt.imshow(hehe3)
plt.show()

'''
img = plt.imread('0008.png')
# print(img.dtype)
#print(img.shape)
#img = img[1650:1750, 900:1000]
#hehe = metodainterpolacjidwuliniowej(img, 120)
#hehe = metodasredniej(img, 80)
#hehe = ditheringzorganizowany(img, 4)
#hehe = ditheringfloydsteinberg(img, palette1)
hehe = metodainterpolacjidwuliniowej(img, 50)
hehe2 = metodainterpolacjidwuliniowej(img, 50)
hehe3 = metodainterpolacjidwuliniowej(img, 50)
hehe = np.clip(hehe, 0, 1)
hehe2 = np.clip(hehe, 0, 1)
hehe3 = np.clip(hehe, 0, 1)
plt.subplot(1,3,1)
plt.title('metoda najblizszego sasiada')
plt.imshow(hehe)
plt.subplot(1,3,2)
plt.title('metoda interpolacji dwuliniowej')
plt.imshow(hehe2)
plt.subplot(1,3,3)
plt.title('metoda sredniej')
plt.imshow(hehe3)
plt.show()
x = hehe
ksize = 1
sobelx = cv2.Sobel(src=x, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)  # Sobel Edge Detection on the X axis

sobely = cv2.Sobel(src=x, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)  # Sobel Edge Detection on the Y axis

sobelxy = cv2.Sobel(src=x, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize)  # Combined X and Y Sobel Edge Detection


xx = cv2.resize(sobelxy, (256, 320))
cv2.imshow('Sobel X Y using Sobel() function', xx)

cv2.waitKey(0)
#plt.imsave('img.png', hehe, cmap='gray')
#plt.imsave('img2.png', hehe2, cmap='gray')
#plt.imsave('img3.png', hehe3, cmap='gray')

#hehe = metodanajblizszegosasiada(img, 120)

print(hehe.dtype)
print(hehe.shape)
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(hehe)
plt.show()

'''
#print(np.around(2.8).astype(np.uint8))


#plt.imsave('img.png', hehe, cmap='gray')
#plt.imshow(img)
#plt.show()

