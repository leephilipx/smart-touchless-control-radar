import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from librosa.feature import mfcc

def get_magnitude(data):
    '''
    Returns the absolute magnitude without normalization as a numpy array (Nsamples, NTS, Nframes)
    '''
    return np.abs(np.swapaxes(data, 1, 2))

def reshape_features(data, type):
    '''
    Reshapes the data features depending on the model type.
    '''
    if type == 'ml':
        return data.reshape(data.shape[0], -1)
    elif type == 'dl':
        if len(data.shape) == 4: return data
        return np.expand_dims(data, axis=-1)

def one_hot_dl(y):
    '''
    One hot encodes the labels or a list of labels.
    '''
    if type(y) is list:
        return [np.eye(np.unique(labels).shape[0])[labels] for labels in y]
    else:
        return np.eye(np.unique(y).shape[0])[y]

def get_stft(radarData):
    f0 = 60e9       	# radar operating frequency
    Fs = 2*f0/1e6       # sampling frequency
    # Unwrap and normalize data
    radar1D = radarData.reshape(-1, 1).squeeze()
    radar1D = radar1D - radar1D.mean()
    # Some parameters to tweak
    overlapPercent = 0.8                                     # Percentage of data overlapping STFT
    NFFT = 4096 * 2   	                                     # FFT point numbers: 2^10=1024 2^12=4096 2^13=8192
    # Doppler window for FFT sidelobe suppression and STFT
    NOverlap = round(NFFT*overlapPercent);                   # Number of overlapping points
    window = signal.windows.taylor(NFFT, nbar=10, sll=80, norm=False)
    dAxis, tAxis, STFT = signal.spectrogram(radar1D, axis=0, fs=Fs, window=window, noverlap=NOverlap, nfft=NFFT, nperseg=NFFT)     
    # Frequency shifting and representation of magnitude in dB
    STFTShift = np.fft.fftshift(STFT, axes=0)                # fftshift means 0 Doppler frequency is at center
    STFTShiftDB = 20 * np.log10(np.abs(STFTShift))           # Represent in dB
    STFTShiftDB = STFTShiftDB - np.max(STFTShiftDB)          # Normalize data to max value (i.e., the max value is 0dB)
    return dAxis, tAxis, STFTShiftDB

def get_mfcc(radarData):
    radar1D = radarData.reshape(-1, 1).squeeze()
    radar1D = radar1D - radar1D.mean()
    return mfcc(np.real(radar1D), sr = 1, n_mfcc = 80)

def get_batch(radarData, mode):
    Nsamples = radarData.shape[0]
    if mode == 'stft':
        return np.array([get_stft(radarData[i, :]) for i in range(Nsamples)])
    elif mode == 'mfcc':
        return np.array([get_mfcc(radarData[i, :]) for i in range(Nsamples)])

def get_stft_plot(index, class_labels):
    fig, axes = plt.subplots(2,2, figsize=(10,40))
    axes = axes.ravel()
    fig.suptitle('Doppler-Time Response (STFT)');
    for i, ax in zip(index, axes):
        x = X[i]
        y = Y[i]
        label = class_labels[y]
        dAxis, tAxis, STFT = get_stft(x)
        dAxis = np.fft.fftshift(dAxis)               # Shift center as 0 Doppler frequency
        ax.pcolormesh(tAxis, dAxis/1e3, STFT, cmap='jet', vmin =-40, vmax = -3)
        ax.set_title(f'{label}');
        # ax.set_xlabel('Time (s)');
        ax.set_ylabel('Doppler (kHz)');
        ax.set_xlim((0,0.5));
        ax.set_ylim((-2,2));
    # plt.colorbar();
    plt.show();

if __name__ == "__main__":

    import radar
    X, Y, class_labels = radar.getTrainData(source_dir='2021_10_13_data')
    print(X.shape, Y.shape, class_labels)
    get_stft_plot(index=(0, 250, 501, 753), class_labels=class_labels)
    # X_STFT = get_mfcc(X)
    # print(X_STFT.shape)
    # X_input = reshape_features(X_STFT, type='dl')
    # print(X_input.shape)
    # y_one_hot = one_hot_dl(Y)
    # print(y_one_hot.shape)