import colorsys
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf
from matplotlib import pyplot as plt
import scipy.fftpack


def energy(frame):
    e = np.sum(frame**2)/frame.shape[0]
    return e


def zcr(frame):
    s = np.sign(frame)
    z = (np.sum(s[:-1]!=s[1:])/frame.shape[0])
    return z


window_time = 20  # ms
window_hop_time = 5  # ms
data, fs = sf.read('50764_1_0.wav')#, dtype=np.float32)
data = data[:, 0]
#plt.plot(np.arange(data.shape[0]), data)
#plt.show()

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
print("p",int(94000/data.shape[0] * frames.shape[1]))
ramka = frames[int(94000/data.shape[0] * frames.shape[1])]
#w = np.ones(wsize,) # okno prostokątne
#w = np.hamming(wsize)
w = np.blackman(wsize)
#w = scipy.signal.windows.blackmanharris(wsize)
print("ramkaa", ramka.shape[0])
#plt.plot(np.linspace(0,ramka.shape[0]/fs,len(ramka)), ramka)
#plt.show()

fsize = 2**9 # 2**8
g = np.max(20 * np.log10(np.abs(scipy.fftpack.fft(w, fsize))))
w_frame = w * frame

plt.figure()
#plt.plot(np.arange(0,fs/2,fs/fsize), 20*np.log10(np.abs(scipy.fftpack.fft(w, fsize)))
w_frame_spec=(20*np.log10( np.abs(scipy.fftpack.fft(w_frame,fsize))))-g
#plt.plot(np.arange(0,fs/2,fs/fsize),w_frame_spec[:fsize//2])
#plt.show()

energyarr = np.empty(0)
zcrarr = np.empty(0)

for xd in range(0, len(frames)):
    zcrarr = np.append(zcrarr, zcr(frames[xd]))
    energyarr = np.append(energyarr, energy(frames[xd]))



plt.plot(np.linspace(0, zcrarr.shape[0], len(zcrarr)), zcrarr)
plt.show()
plt.plot(np.linspace(0, energyarr.shape[0], len(energyarr)), energyarr)
plt.show()



