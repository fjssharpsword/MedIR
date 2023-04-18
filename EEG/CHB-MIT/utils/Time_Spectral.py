import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from matplotlib.pylab import mpl
 

def get_fft_values():
    # sampling rate
    sr = 2000
    # sampling interval
    ts = 1.0/sr
    t = np.arange(0,1,ts)

    freq = 1.
    x = 3*np.sin(2*np.pi*freq*t)

    freq = 4
    x += np.sin(2*np.pi*freq*t)

    freq = 7   
    x += 0.5* np.sin(2*np.pi*freq*t)

    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    freq = n/T 

    plt.figure(figsize = (12, 6))
    plt.subplot(121)

    plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 10)

    plt.subplot(122)
    plt.plot(t, ifft(X), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    plt.savefig('/data/pycode/MedIR/EEG/CHB-MIT/imgs/time_spectral.png', bbox_inches='tight')

def calculate_freq():
    # assign data
    x = np.array([1,2,1,0,1,2,1,0])
    # compute DFT with optimized FFT
    w = np.fft.fft(x)
    # compute frequency associated with coefficients
    freqs = np.fft.fftfreq(n=len(x), d=1.0)

    # extract frequencies associated with FFT values
    for coef, freq in zip(w, freqs):
        if coef:
            print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef, f=freq))

def output_specgram():
    # sampling rate
    sr = 256
    # sampling interval
    ts = 1.0/sr
    #t = np.arange(0,2,ts)
    t = np.arange(0,20,ts)

    freq = 1.
    x = 3*np.sin(2*np.pi*freq*t)

    freq = 4
    x += np.sin(2*np.pi*freq*t)

    freq = 7   
    x += 0.5* np.sin(2*np.pi*freq*t)

    #Pxx, freqs, bins, im = plt.specgram(x, NFFT=256, Fs=40000000, noverlap=254)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=2560, Fs=2, noverlap=2558)

    print('{}--{}'.format(Pxx.shape[0], Pxx.shape[1]))

if __name__ == "__main__":
    #get_fft_values()
    #calculate_freq()
    output_specgram()