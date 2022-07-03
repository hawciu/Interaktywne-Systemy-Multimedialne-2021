import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import signal
import sounddevice as sd
import soundfile as sf
from scipy.interpolate import interp1d


notes = {
    'R': 0,
    'RR': 0,
    'C0': 16.35,
    'C0#': 17.32,
    'D0': 18.35,
    'D0#': 19.45,
    'E0': 20.6,
    'F0': 21.83,
    'F0#': 23.12,
    'G0': 24.5,
    'G0#': 25.96,
    'A0': 27.5,
    'A0#': 29.14,
    'B0': 30.87,
    'C1': 32.7,
    'C1#': 34.65,
    'D1': 36.71,
    'D1#': 38.89,
    'E1': 41.2,
    'F1': 43.65,
    'F1#': 46.25,
    'G1': 49,
    'G1#': 51.91,
    'A1': 55,
    'A1#': 58.27,
    'B1': 61.74,
    'C2': 65.41,
    'C2#': 69.3,
    'D2': 73.42,
    'D2#': 77.78,
    'E2': 82.41,
    'F2': 87.31,
    'F2#': 92.5,
    'G2': 98,
    'G2#': 103.83,
    'A2': 110,
    'A2#': 116.54,
    'B2': 123.47,
    'C3': 130.81,
    'C3#': 138.59,
    'D3': 146.83,
    'D3#': 155.56,
    'E3': 164.81,
    'F3': 174.61,
    'F3#': 185,
    'G3': 196,
    'G3#': 207.65,
    'A3': 220,
    'A3#': 233.08,
    'B3': 246.94,
    'C4': 261.63,
    'C4#': 277.18,
    'D4': 293.66,
    'D4#': 311.13,
    'E4': 329.63,
    'F4': 349.23,
    'F4#': 369.99,
    'G4': 392,
    'G4#': 415.3,
    'A4': 440,
    'A4#': 466.16,
    'B4': 493.88,
    'C5': 523.25,
    'C5#': 554.37,
    'D5': 587.33,
    'D5#': 622.25,
    'E5': 659.25,
    'F5': 698.46,
    'F5#': 739.99,
    'G5': 783.99,
    'G5#': 830.61,
    'A5': 880,
    'A5#': 932.33,
    'B5': 987.77,
    'C6': 1046.5,
    'C6#': 1108.73,
    'D6': 1174.66,
    'D6#': 1244.51,
    'E6': 1318.51,
    'F6': 1396.91,
    'F6#': 1479.98,
    'G6': 1567.98,
    'G6#': 1661.22,
    'A6': 1760,
    'A6#': 1864.66,
    'B6': 1975.53,
    'C7': 2093,
    'C7#': 2217.46,
    'D7': 2349.32,
    'D7#': 2489.02,
    'E7': 2637.02,
    'F7': 2793.83,
    'F7#': 2959.96,
    'G7': 3135.96,
    'G7#': 3322.44,
    'A7': 3520,
    'A7#': 3729.31,
    'B7': 3951.07,
    'C8': 4186.01,
    'C8#': 4434.92,
    'D8': 4698.63,
    'D8#': 4978.03,
    'E8': 5274.04,
    'F8': 5587.65,
    'F8#': 5919.91,
    'G8': 6271.93,
    'G8#': 6644.88,
    'A8': 7040,
    'A8#': 7458.62,
    'B8': 7902.13,
}


def generujadsr(lengthin):
    if lengthin < 3000:
        mid = int(lengthin/2)
        result = np.linspace(-1, -0.9, mid)
        result = np.append(result, np.linspace(-0.9, -1, lengthin-mid))
        #plt.plot(result)
        #plt.show()
    else:
        hodl = 0.5
        sustain = 5
        peak = 1000
        drop = 1500
        dropend = int((lengthin-peak-drop) * 0.8)#int((lengthin - drop)/sustain + drop)
        end = int(lengthin)
        result = np.linspace(0, 1, peak)
        result = np.append(result, np.linspace(1, hodl, (drop-peak)))
        result = np.append(result, np.zeros((dropend-drop)) + hodl)
        result = np.append(result, np.linspace(hodl, 0, (end - dropend)))
    #plt.plot(np.arange(0, lengthin), result)
    #plt.show()
    return result


def generujDzwiek(typ, hzin, durationin):
    if typ == 1:  # sinusoidalna
        result = np.sin(2 * np.pi * np.arange(fs * durationin) * hzin/fs)
    if typ == 2:  # prostokatna
        result = scipy.signal.square(2 * np.pi * np.arange(fs * durationin) * hzin/fs)
    if typ == 3:  # trojkatna (triangle is absolute value of sawtooth)
        result = np.abs(scipy.signal.sawtooth(2 * np.pi * np.arange(fs * durationin) * hzin/fs)) * 2 - 1
    if typ == 4:  # pila
        result = scipy.signal.sawtooth(2 * np.pi * np.arange(fs * durationin) * hzin/fs)
    if typ == 5:
        t = (2 * np.pi * np.arange(fs * durationin) * hzin / fs)
        w = 1
        ww = -0.0015
        result = np.sin(w * t) * np.exp(ww * w * t)
        result += np.sin(2 * w * t) * np.exp(ww * w * t)
        result += np.sin(4 * w * t) * np.exp(ww * w * t)
        #result += np.sin(6 * w * t) * np.exp(ww * w * t)
        #result += np.sin(8 * w * t) * np.exp(ww * w * t)
        if max(result) != 0:
            result = result / max(np.abs(result))
    if typ == 6:
        t = (2 * np.pi * np.arange(fs * durationin) * hzin / fs)
        w = 0.5
        ww = -0.005
        result = np.sin(w * t) * np.exp(ww * w * t)
        result += np.sin(2 * w * t) * np.exp(ww * w * t)
        result += np.sin(4 * w * t) * np.exp(ww * w * t)
        # result += np.sin(6 * w * t) * np.exp(ww * w * t)
        # result += np.sin(8 * w * t) * np.exp(ww * w * t)
        if max(result) != 0:
            result = result / max(np.abs(result))

        '''
        tt = 2 * np.pi * np.arange(fs * durationin) * hzin / fs
        ww = -0.0025
        result = np.sin(tt) * np.exp(ww * tt)
        result += np.sin(2 * tt) / 2 * np.exp(ww * tt) / 2
        result += np.sin(4 * tt) / 2 * np.exp(ww * tt) / 4
        result += np.sin(8 * tt) / 2 * np.exp(ww * tt) / 8
        result += np.sin(16 * tt) / 2 * np.exp(ww * tt) / 16
        '''
    #dodaj atak
    #result = result * generujadsr(len(result))
    return result


def grajmuzyko(tytul):
    with open(tytul) as f:
        # lines = f.readlines()
        lines = f.read().splitlines()

    bpm = int(lines[0])
    lines.pop(0)
    noteduration = 120/bpm
    utworout = np.empty(0)
    instrument = 1
    for j in lines:
        print("l", 1)
        utworpart = np.empty(0)
        currentnotes = j.split()
        instrument = int(currentnotes[0])
        currentnotes.pop(0)
        for i in currentnotes:
            data = generujDzwiek(instrument, notes[i.split('-')[0]], noteduration / int(i.split('-')[1]))
            data = data * generujadsr(len(data))
            utworpart = np.append(utworpart, data)
        # sd.play(utworpart, fs)
        # status = sd.wait()
        if len(utworout) == 0:
            utworout = np.append(utworout, utworpart)
        else:
            utworout = utworout + utworpart
    utworout = utworout / max(utworout)
    print(max(utworout), min(utworout))
    return utworout


def modulacjaczestotliwosciiamplitudy():
    plt.subplot(4, 1, 1)
    a = generujDzwiek(1, 40, 0.8)
    plt.plot(np.arange(0, a.shape[0]), a)
    plt.subplot(4, 1, 1)
    aa = generujDzwiek(1, 5, 0.8)
    aaa = aa * 0.25 + 0.75
    plt.plot(np.arange(0, aaa.shape[0]), aaa)
    plt.title("fala bazowa")
    plt.xlabel("czas")
    plt.ylabel("amplituda")
    plt.text(19000, 0.8, 'modulacja amplitudy', color='orange')

    plt.subplot(4, 1, 2)
    aaaa = a * aaa
    plt.plot(np.arange(0, aaaa.shape[0]), aaaa)
    plt.subplot(4, 1, 2)
    plt.plot(np.arange(0, aaa.shape[0]), aaa)
    plt.title("modulacja amplitudy fali bazowej")
    plt.xlabel("czas")
    plt.ylabel("amplituda")
    plt.text(19000, 0.8, 'modulacja amplitudy', color='orange')

    plt.subplot(4, 1, 3)
    #a * np.sin(2 * np.pi * np.arange(fs * durationin) * hzin/fs)
    baza = generujDzwiek(1, 40, 0.8)
    modulacja = generujDzwiek(1, 10, 0.8)
    aaaaa = np.sin(2 * np.pi * np.cumsum((10*modulacja+40)/fs))
    plt.plot(np.arange(0, aaaaa.shape[0]), aaaaa)
    plt.subplot(4, 1, 3)
    plt.plot(np.arange(0, modulacja.shape[0]), modulacja * 0.2 + 0.5)
    plt.title("modulacja częstotliwości fali bazowej")
    plt.xlabel("czas")
    plt.ylabel("amplituda")
    plt.text(19000, 0.6, 'modulacja częstotliwości', color='orange')

    plt.subplot(4, 1, 4)
    modamp = 0.25 * (generujDzwiek(1, 5, 0.8)) + 0.75
    aaaaaa = aaaaa * modamp
    plt.plot(np.arange(0, aaaaaa.shape[0]), aaaaaa)
    plt.plot(np.arange(0, modamp.shape[0]), modamp)
    plt.title("modulacja częstotliwości i amplitudy fali bazowej")
    plt.xlabel("czas")
    plt.ylabel("amplituda")
    plt.text(19000, 0.8, 'modulacja amplitudy', color='orange')

    plt.subplots_adjust(hspace=0.7)
    plt.show()
    return True


def maksima_widma(widmo, prog, rozmiar_okna, fs):
    ponad = (widmo >= prog).astype(int)
    pochodna = np.diff(ponad)
    poczatki = np.where(pochodna == 1)[0] + 1
    konce = np.where(pochodna == -1)[0] + 1
    maksima = []
    for poczatek, koniec in zip(poczatki, konce):
        p = np.argmax(widmo[poczatek:koniec]) + poczatek
        a, b, c = widmo[p - 1:p + 2]
        k = 0.5 * (a - c) / (a - 2 * b + c)
        maksima.append((p + k) * fs / rozmiar_okna)
    return maksima


fs = 24000  ########################################

#utwor = grajmuzyko('buk sie rodzi.txt')
utwor = grajmuzyko('betlejem.txt')
sd.play(utwor, fs)
status = sd.wait()
#sf.write('bog sie rodzi.wav', utwor, fs)


# szukanie peakow harmonizacji
if False:
    tt = 2 * np.pi * np.arange(fs * 1) * 440 / fs
    ww = 1
    tt = 2 * np.pi * np.arange(fs * 1) * 440 / fs
    ww = -0.0025
    result2 = np.sin(tt) * np.exp(ww * tt)
    result2 += np.sin(2 * tt) / 2 * np.exp(ww * tt) / 2
    result2 += np.sin(4 * tt) / 2 * np.exp(ww * tt) / 4
    result2 += np.sin(8 * tt) / 2 * np.exp(ww * tt) / 8
    result2 += np.sin(16 * tt) / 2 * np.exp(ww * tt) / 16
    plt.plot(result2)
    #plt.show()
    yf = scipy.fftpack.fft(result2,fs)
    plt.plot(np.arange(0,fs/2,fs/fs),20*np.log10( np.abs(yf[:fs//2])))
    #plt.show()

    fragment = result2[10000:12048]
    fragment = fragment / np.max(np.abs(fragment))
    widmo = 20 * np.log10(np.abs(np.fft.rfft(fragment * np.hamming(2048))) / 1024)
    f = np.fft.rfftfreq(2048, 1 / fs)
    plt.plot(f, widmo)
    plt.xlim(0, 10000)
    plt.ylim(-90, 0)
    plt.xlabel('częstotliwość [Hz]')
    plt.ylabel('amplituda widma [dB]')
    plt.title('Widmo dźwięku')
    plt.show()


    maksima = maksima_widma(widmo, -45, 2048, fs)
    for m in maksima:
        print('f = {:8.3f}, współczynnik = {:5.2f}'.format(m, m / maksima[0]))


