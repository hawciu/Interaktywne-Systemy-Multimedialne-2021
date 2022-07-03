import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt


A = 87.6
u = 256

def alawkompresja(datain):
    result = np.empty(0)
    datain_copy = datain.copy()
    # for i in datain:
    for i in range(len(datain_copy)):
        if np.abs(datain_copy[i]) < (1 / A):
            result = np.append(result, (np.sign(datain_copy[i]) * (A * np.abs(datain_copy[i]) / (1 + np.log(A)))))

        else:
            result = np.append(result, (np.sign(datain_copy[i]) *((1 + np.log(A * np.abs(datain_copy[i]))) / (1 + np.log(A)))))
    return result


def alawdekompresja(datain):
    result  = np.empty(0)
    datain_copy = datain.copy()
    for i in range(len(datain_copy)):
        if np.abs(datain_copy[i]) < (1 / (1 + np.log(A))):
            result = np.append(result, np.sign(datain_copy[i])*((np.abs(datain_copy[i]) * (1 + np.log(A))) / A))

        else:
            result = np.append(result, np.sign(datain_copy[i]) * ((np.exp(np.abs(datain_copy[i])*(1 + np.log(A)) - 1)) / A))
    return result


def ulawkompresja(datain):
    datain_copy = datain.copy()
    result = np.empty(0)
    for i in range(len(datain)):
        #if np.abs(datain[i]) < (1 / A):
        result = np.append(result, np.sign(datain_copy[i]) * (np.log(1 + u*np.abs(datain_copy[i]))) / (np.log(1+u)) )
        print("kom:", result)

    return result

def ulawdekompresja(datain):
    result = np.empty(0)
    datain_copy = datain.copy()
    for i in range(len(datain)):
        tmp = np.sign(datain_copy[i]) * (1/u) * ( (np.pow((1+u), np.abs(datain_copy[i])) - 1))
        result = np.append(result, tmp)

    return result


def kwantyzacja(datain, bity):
    if np.issubdtype(datain[0].dtype, np.integer):
        print("int")
        palette = np.linspace(1, 2**bity, 2**bity).astype(int)
    else:
        print("float")
        palette = np.linspace(-1, 1, 2**bity)
        print(palette)
    datain_copy = datain.copy()
    result = np.empty(0)
    for i in range(len(datain_copy)):
        value1 = datain_copy[i]
        value = palette[np.argmin(np.absolute(palette - value1))]
        result = np.append(result, value)

    return result



def kompresjaDCPM(datain, bity): # calkowite
    datain_copy = datain.copy()
    E = datain_copy[0]
    result = np.empty(0)
    for i in range(len(datain_copy)):
        Y = kwantyzacja(np.array([datain_copy[i] - E]), bity)
        E = E + Y
        result = np.append(result, Y)
    return True



#data, fs = sf.read('sing_high2.wav', dtype=np.int32)
data, fs = sf.read('sin_60Hz.wav')
#data = data * 2147483647
print(data)
# data2 =Alaw_kompresja(data)
#sd.play(data2, fs)
#status = sd.wait()
data22 = alawkompresja(data)
data2 = kwantyzacja(data22, 8)
data3 = alawdekompresja(data2)

plt.plot(np.linspace(-1, 1, 1000), data2[0:1000])
plt.show()



'''
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data.shape[0])/fs,data)

plt.subplot(2,1,2)
plt.plot(np.arange(0,data2.shape[0])/fs,data2)


plt.show()


'''







