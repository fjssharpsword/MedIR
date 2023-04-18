import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    get_fft_values()