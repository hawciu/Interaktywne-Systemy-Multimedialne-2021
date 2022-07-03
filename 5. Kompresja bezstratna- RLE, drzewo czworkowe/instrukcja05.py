import numpy as np
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import sys


class branch:
    children = []
    color = []
    isleaf = False

    def __init__(self, x, y, datain):
        self.x = x
        self.y = y
        self.sizex = datain.shape[0]
        self.sizey = datain.shape[1]
        if datain.shape == (1, 1, 3):
            self.color = np.array([datain[0, 0, 0], datain[0, 0, 1], datain[0, 0, 2]])
            self.isleaf = True
        elif pixelcheck(datain):
            self.color = np.array([datain[0, 0, 0], datain[0, 0, 1], datain[0, 0, 2]])
            self.isleaf = True
        else:
            if self.sizex == 1:
                self.children = np.array([
                    branch(x, y,
                           datain[0:self.sizex, 0:int(self.sizey/2)]),
                    branch(x, y + int(self.sizey/2),
                           datain[0:self.sizex, int(self.sizey/2):self.sizey])
                ])

            elif self.sizey == 1:
                self.children = np.array([
                    branch(x, y,
                           datain[0:int(self.sizex/2), 0:self.sizey]),
                    branch(x + int(self.sizex/2), y,
                           datain[int(self.sizex/2):self.sizex, int(self.sizey/2):self.sizey])
                ])

            else:
                self.children = np.array([
                    branch(x, y,
                           datain[0:int(self.sizex/2), 0:int(self.sizey/2)]),
                    branch(x + int(self.sizex/2), y,
                           datain[int(self.sizex/2):self.sizex, 0:int(self.sizey/2)]),
                    branch(x, y + int(self.sizey/2),
                           datain[0:int(self.sizex/2), int(self.sizey/2):self.sizey]),
                    branch(x + int(self.sizex/2), y + int(self.sizey/2),
                           datain[int(self.sizex/2):self.sizex, int(self.sizey/2):self.sizey])
                ])



def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj,np.ndarray):
        size += obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def koderRLE2(datain):
    result = np.zeros(datain.shape[0]*datain.shape[1]*4 + 2)
    result[0] = datain.shape[0]
    result[1] = datain.shape[1]
    datain = datain.flatten()
    result[2] = 1
    result[3] = datain[0]
    result[4] = datain[1]
    result[5] = datain[2]
    length = 6
    for i in range(3, len(datain), 3):
        # jesli aktualny jest ten sam co poprzedni
        if np.array_equal([datain[i - 3], datain[i - 2], datain[i - 1]], [datain[i], datain[i + 1], datain[i + 2]]):
            # dodaj count w result + 1
            result[length - 4] = result[length - 4] + 1
        # jesli aktualny jest inny, dopisz sie z count 1
        else:
            result[length] = 1
            result[length + 1] = datain[i]
            result[length + 2] = datain[i + 1]
            result[length + 3] = datain[i + 2]
            length = length + 4
    result = np.resize(result, length)
    return result.astype(int)


def dekoderRLE(datain):
    result = np.zeros(datain[0] * datain[1] * 3)
    licznik = 0
    for i in range(2, len(datain), 4):
        for j in range(datain[i]):
            result[licznik*3] = datain[i + 1]
            result[licznik*3 + 1] = datain[i + 2]
            result[licznik*3 + 2] = datain[i + 3]
            licznik = licznik + 1
    result = result.reshape(datain[0], datain[1], 3)
    return result.astype(int)


def pixelcheck(datain):
    if np.max(datain[:, :, 0]) == np.min(datain[:, :, 0]):
        if np.max(datain[:, :, 1]) == np.min(datain[:, :, 1]):
            if np.max(datain[:, :, 2]) == np.min(datain[:, :, 2]):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def koderDrzewo(datain):
    x = branch(0, 0, datain)
    return [datain.shape, x]


def decodebranch(datain):
    if datain.isleaf:
        tmp = np.array([datain.x, datain.y, datain.sizex, datain.sizey, datain.color[0], datain.color[1], datain.color[2]])
    else:
        tmp = np.empty(0);
        for i in range(len(datain.children)):
            tmp = np.append(tmp, decodebranch(datain.children[i])).astype(int)
    return tmp


def dekoderDrzewo(datain):
    resultimage = np.zeros(datain[0][0] * datain[0][1] * 3).reshape(datain[0][0], datain[0][1], 3)
    result = decodebranch(datain[1])

    # [datain.x, datain.y, datain.sizex, datain.sizey, datain.color[0], datain.color[1], datain.color[2]]
    for i in range(0, len(result), 7):
        for x in range(result[i + 2]):
            for y in range(result[i + 3]):
                resultimage[result[i] + x][result[i + 1] + y][0] = result[i + 4]
                resultimage[result[i] + x][result[i + 1] + y][1] = result[i + 5]
                resultimage[result[i] + x][result[i + 1] + y][2] = result[i + 6]

    return resultimage.astype(int)


img = plt.imread('2.jpg')
print(img.shape)
img1 = koderRLE2(img)
#img1 = koderDrzewo(img)
print("dekoder")
img2 = dekoderRLE(img1)
#img2 = dekoderDrzewo(img1)


print(get_size(img))
print(get_size(img1))
# print(np.array_equal(img, img2))
print("stopien kompresji:", get_size(img)/get_size(img1), ",", 100/(get_size(img)/get_size(img1)), "%")
if np.array_equal(img, img2):
    print("Obraz oryginalny i wynik dekompresji maja te same wartości.")
else:
    print("Obraz oryginalny i wynik dekompresji różnią się.")

plt.imshow(img2)
plt.show()


'''
start = time.time()
img1 = koderRLE2(img).astype(int)
img2 = dekoderRLE(img1).astype(int)
end = time.time()
#for i in range(0, len(img1), 4):
#    print(img1[i], img1[i+1], img1[i+2], img[i+3])
size_img = get_size(img)
size_img1 = get_size(img1)
print("rozmiar oryginalu:", size_img, " rozmiar po kompresji:", size_img1, " stopien kompresji:", size_img/size_img1)
print("czas:", end - start)
plt.imshow(img2)
plt.show()
'''
