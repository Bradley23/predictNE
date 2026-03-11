import numpy as np
from scipy.signal import butter, filtfilt

def calc_connectivity(signal, window_size=100):
    N, T, reg = signal.shape
    n_windows = T - window_size + 1

    triu_indices = np.triu_indices(reg, k=1)

    connectivity = np.empty((N, n_windows, len(triu_indices[0])))

    for n in range(N):
        sig = signal[n]

        for i in range(n_windows):
            window = sig[i:i+window_size, :].copy()
            window -= window.mean(axis=0, keepdims=True)

            window /= window.std(axis=0, ddof=1, keepdims=True)

            conn = (window.T @ window) / (window_size - 1)
            connectivity[n, i] = conn[triu_indices[0], triu_indices[1]]    

    return connectivity

def window_clip(signal, window_size=100):
    T = signal.shape[1]
    n_windows = T - window_size + 1

    T_indices = range(window_size // 2, T - window_size // 2 + 1)

    return signal[:, T_indices]

def bpf(signal, freq=[0, 0.5], fs=10):
    nyq = fs / 2
    low = freq[0] / nyq
    high = freq[1] / nyq
    
    if freq[0] == 0:
        btype = 'low'
        freq = high
    elif freq[1] == nyq:
        btype = 'high'
        freq = low
    else:
        btype = 'band'
        freq = [low, high]

    b, a = butter(6, freq, btype=btype)
    filtered_data = filtfilt(b, a, signal, axis=1)

    return filtered_data
