import colorsys
import numpy as np
import scipy.ndimage
from cv2 import cv2
from matplotlib import pyplot as plt


def kwantyzacjah(valuein):
    skala = 255/360
    r = 0
    if valuein >= 20 * skala:
        r = 1
        if valuein >= 40 * skala:
            r = 2
            if valuein >= 75* skala:
                r = 3
                if valuein >= 155 * skala:
                    r = 4
                    if valuein >= 190 * skala:
                        r = 5
                        if valuein >= 270 * skala:
                            r = 6
                            if valuein >= 295 * skala:
                                r = 7
                                if valuein >= 316 * skala:
                                    r = 0
    return r


def kwantyzacjavs(valuein):
    skala = 255/100
    r = 0
    if valuein >= 20 * skala:
        r = 1
        if valuein >= 70 * skala:
            r = 2
            if valuein >= 100 * skala:
                r = 0
    return r


def kwantyzacjaobrazu(imgin):
    result = np.zeros((imgin.shape[0], imgin.shape[1], 3))
    for x in range(imgin.shape[0]):
        for y in range(imgin.shape[1]):
            result[x][y][0] = kwantyzacjah(imgin[x][y][0])
            result[x][y][1] = kwantyzacjavs(imgin[x][y][1])
            result[x][y][2] = kwantyzacjavs(imgin[x][y][2])
    return result.astype(int)


def tworzhistogram(imgin):
    result = np.zeros((8, 3, 3)).astype(int)
    for x in range(imgin.shape[0]):
        for y in range(imgin.shape[1]):
            result[imgin[x][y][0], imgin[x][y][1], imgin[x][y][2]] += 1
    return result


def linearyzacjahistogramu(hisin):
    result = []
    for h in range(hisin.shape[0]):
        for s in range(hisin.shape[1]):
            for v in range(hisin.shape[2]):
                result.append(hisin[h][s][v])
    return result


def uproszczonydcd(namein, n=8):
    imgin = plt.imread(namein)
    imgin = cv2.cvtColor(imgin, cv2.COLOR_RGB2HSV)
    imgin = kwantyzacjaobrazu(imgin)
    histogram3d = tworzhistogram(imgin)
    histogram1d = linearyzacjahistogramu(histogram3d)
    listakolorow = np.zeros((n, 2)).astype(int)  # ilosc, indeks(kolor)
    tmphist = histogram1d.copy()
    #print(tmphist)
    for i in range(n):
        tmp = max(tmphist)
        #print("tmp", tmp)
        listakolorow[i][0] = tmp  # kolor
        tmp2 = tmphist.index(tmp)
        #print("tmp2", tmp2)
        listakolorow[i][1] = tmp2
        tmphist[tmp2] = 0
    #print("lista kolorow", listakolorow)
    listadominujacychkolorow = np.zeros((n, 2)).astype(float)  # kolor, %
    for i in range(n):
        listadominujacychkolorow[i][0] = round(listakolorow[i][1], 0)
        listadominujacychkolorow[i][1] = round(listakolorow[i][0]/sum(histogram1d), 3)
    #print("lista dominujacych kolorow", listadominujacychkolorow)
    # zerowanie histogramu i wstawienie dominujacych
    tmphist = np.zeros(72)
    for i in range(n):
        tmphist[listakolorow[i][1]] = listakolorow[i][0] / sum(listakolorow[:, 0])
    #print("wyzerowany histogram", tmphist, sum(tmphist))
    return listadominujacychkolorow, tmphist


def uproszczonyehd(namein):
    imgin = plt.imread(namein)
    imggs = np.dot(imgin[..., :3], [0.2989, 0.5870, 0.1140])
    plt.imsave('11.png', imggs, cmap='Greys_r')

    histogram80 = np.zeros(80)
    x = int(imggs.shape[0]/4)
    y = int(imggs.shape[1]/4)
    for i in range(4):
        for j in range(4):  # 16 blokow
            histogram5 = np.zeros(5)
            for ii in range(0, x, 2):
                for jj in range(0, y, 2):  # po pikselach bloki 2x2
                    dwablok = imggs[i*4+ii:i*4+ii+2, j*4+jj:j*4+jj+2]
                    histogram5[0] = abs(np.sum(dwablok * np.array([[1, -1], [1, -1]])))  # 90 stopni
                    histogram5[1] = abs(np.sum(dwablok * np.array([[1, 1], [-1, -1]])))  # 0 stopni
                    histogram5[2] = abs(np.sum(dwablok * np.array([[2**(1/2), 0], [0, -1*2**(1/2)]])))  # 45 stopni
                    histogram5[3] = abs(np.sum(dwablok * np.array([[0,2**(1/2)], [-1*2**(1/2), 0]])))  # 135 stopni
                    histogram5[4] = abs(np.sum(dwablok * np.array([[2, -2], [-2, 2]])))  # bez okreslonego kierunku
                    # histogram80[i*4*5 + histogram5.index(max(histogram5))] += 1
                    histogram80[i * 4 * 5 + np.argmax(histogram5)] += 1
    return histogram80.astype(int)


def podobienstwo(hist1, hist2):
    result = 0
    for hist in range(len(hist1)):
        if (float(hist1[hist])) > 0:
            result += ((float(hist1[hist]) - float(hist2[hist]))**2) / float(hist1[hist])
    return result


#img3 = cv2.cvtColor(img2, cv2.COLOR_HSV2RGB)


#uproszczonydcd(img)
suffiksy = ['', '_20', '_50', '_70', '_90', '_180', '_270', '_lustro', '_negatyw', '1', '2', '3', '4', '5']
#obrazy = ['dom', 'drzewo', 'pies', 'samochod', 'zamek']
obrazy = ['dom']

# zapis wynikow do plikow
if False:
    for i in obrazy:
        for j in suffiksy:
            nazwa = i+j
            print(nazwa)
            aa, a = uproszczonydcd('baza obrazow/'+nazwa+'.jpg')
            with open('dcd/'+nazwa+'.txt', 'w') as f:
                for x in a:
                    f.write(str(x))
                    f.write(' ')
            a = uproszczonyehd('baza obrazow/'+nazwa+'.jpg')
            with open('ehd/'+nazwa+'.txt', 'w') as f:
                for x in a:
                    f.write(str(x))
                    f.write(' ')

# porownanie wynikow

if True:
    with open('dcd/dom.txt') as f:
        baza = f.read().split()
    for i in obrazy:
        for j in suffiksy:
            nazwa = i+j
            print(nazwa)
            with open('dcd/'+nazwa+'.txt') as f:
                lines = f.read().split()
            print(podobienstwo(baza, lines))

