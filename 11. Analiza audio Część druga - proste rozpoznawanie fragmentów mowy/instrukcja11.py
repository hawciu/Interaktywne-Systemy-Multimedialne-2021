import colorsys
import numpy as np
import scipy.ndimage
import sounddevice as sd
import soundfile as sf
from cv2 import cv2
from matplotlib import pyplot as plt
import scipy.fftpack
import librosa
from scipy.signal import lfilter


def get_F0_autocorr(framein, fsin):
    corr = np.correlate(framein, framein, mode='full')
    corr = corr[corr.size//2:]
    d = np.diff(corr)
    start = np.nonzero(d > 0)[0][0]
    peak = np.argmax(corr[start:]) + start
    return fsin / peak


def get_Formants(frame,fs):
    ncoeff = np.ceil(2 + fs / 1000).astype(int)
    x1 =  lfilter([1], [1., 0.63], frame)
    A = librosa.lpc(x1, int(ncoeff))
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    frqs = sorted(angz * (fs / (2 * np.pi)))
    if frqs[0]<1:
        for i in range(0,len(frqs)):
            if frqs[i]>=1:
                break
        return frqs[i:]
    return frqs


data, fs = sf.read('41375_2_1.wav')#, dtype=np.float32)
data = data[:, 0]
window_time = 20  # ms
window_hop_time = 5  # ms
plt.plot(np.arange(data.shape[0]), data)
plt.show()

window_samples = np.round(window_time / 1000 * fs).astype(int)
wsize = window_samples

window_hops = np.round(window_hop_time / 1000 * fs).astype(int)
print("ws", window_samples, "wh", window_hops)

frames = np.empty([0, window_samples])

for ix in range(0, data.shape[0] - window_samples + 1, window_hops):
    frame = data[ix:ix + window_samples]
    # print(ix/fs, frame.shape[0]/fs)
    # print(ix/fs, frame.shape[0])
    frames = np.concatenate((frames, [frame]), axis=0)

print(data.shape)
print(frames.shape)
# ramka = frames[int(35000/window_samples * window_hops)]
ramka = frames[int(33500/data.shape[0] * frames.shape[1])]

a = get_F0_autocorr(ramka, fs)
print(a)

w = np.hamming(wsize)
fsize = 2**8  # 2**8
g = np.max(20 * np.log10(np.abs(scipy.fftpack.fft(w, fsize))))
w_frame = w * ramka

plt.figure()
plt.subplot(3,1,1)
plt.plot(np.arange(0,ramka.shape[0])/fs,ramka)
plt.title("ramka")

plt.subplot(3,1,2)
yf = scipy.fftpack.fft(ramka,fsize)
plt.plot(np.arange(0,fs,fs/fsize),np.abs(yf))
plt.title("widmo całe")
plt.subplot(3,1,3)
yf = scipy.fftpack.fft(ramka,fsize)
plt.plot(np.arange(0,fs,fs/fsize),np.abs(yf))
plt.title("widmo przybliżone na pik")
#yf = scipy.fftpack.fft(w,fsize)
#plt.plot(np.arange(0,fs,fs/fsize),np.abs(yf))
plt.show()

aa = get_Formants(ramka, fs)
print(aa)
