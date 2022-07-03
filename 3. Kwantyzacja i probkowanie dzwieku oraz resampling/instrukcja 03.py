#Kwantyzacja i próbkowanie dźwięku oraz resampling
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import sounddevice as sd
import soundfile as sf
from scipy.interpolate import interp1d


def zmienrozdzielczosc(iloscbitow, datain):
    for i in range(len(datain)): #redukcja bitow na wybrana ilosc
        if datain[i] > (2**iloscbitow/2-1):
            datain[i] = 2**iloscbitow/2-1
        if datain[i] < (-2**iloscbitow/2):
            datain[i] = -2**iloscbitow/2
#        datain[i] = datain[i]/(2**32)*(2**iloscbitow)
#        if iloscbitow < 16:#jesli mniej niz 16 to nie odtworzy dzwieku, wiec wzrost do 16tki
#            datain[i] = datain[i]*((2**16)/(2**iloscbitow))
    return datain


def decymacja(interwal, datain):
    i = 0
    dataout = []
    while i < (len(datain)):
        dataout.append(datain[i])
        i = i+interwal
    dataout = np.array(dataout)  # z listy na numpy array ponoc
    return dataout


def interpolacja(typ, fs, fs2, datain):  # 1 liniowa, 2 nieliniowa, khz docelowe/pobrane
    length = len(datain) / fs
    #timesbase = np.linspace(0, (len(datain))/fs, len(datain))
    x = np.linspace(0, length, len(datain))
    x1 = np.linspace(0, length, int((len(datain))*fs2/fs))

    if typ == 1:
        metode_lin = interp1d(x, datain)
        y_lin = metode_lin(x1)
        return y_lin

    if typ == 2:
        metode_nonlin = interp1d(x, datain, kind='cubic')
        y_lin = metode_nonlin(x1)
        return y_lin



data, fs = sf.read('sing_high2.wav', dtype=np.int32)
print(fs)
sd.play(data, fs)
status = sd.wait()
print (data)
#print(data)
fs2 = 48000
#data2 = interpolacja(1, fs, fs2, data)
data2 = zmienrozdzielczosc(16, data)
# data2 = decymacja(7, data)
#print(data2)

sd.play(data2, fs2)
status = sd.wait()

'''
plt.figure()
plt.subplot(4,1,1)
plt.plot(np.arange(0,data2.shape[0])/fs2,data2)
plt.title('widmo bez modyfikacji')

plt.subplot(4,1,2)
plt.tight_layout(pad=1.0)
yf = scipy.fftpack.fft(data2)
plt.plot(np.arange(0,fs2,1.0*fs2/(yf.size)),np.abs(yf))
plt.title('widmo ze zmniejszonym do 256 rozmiarem', pad=10)

fsize=2**8

plt.subplot(4,1,3)
yf = scipy.fftpack.fft(data2,fsize)
plt.plot(np.arange(0,fs2/2,fs2/fsize),np.abs(yf[:fsize//2]))
plt.title('połowa widma ze zmniejszonym do 256 rozmiarem')

plt.subplot(4,1,4)
yf = scipy.fftpack.fft(data2,fsize)
plt.plot(np.arange(0,fs2/2,fs2/fsize),20*np.log10( np.abs(yf[:fsize//2])))
plt.title('połowa widma ze zmniejszonym do 256 rozmiarem wyświetlona w dB')


plt.show()

'''






