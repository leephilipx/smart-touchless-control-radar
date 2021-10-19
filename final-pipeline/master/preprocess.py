import numpy as np
from scipy import signal
from cv2 import resize
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
    dAxis, tAxis, STFT = signal.spectrogram(radar1D, axis=0, fs=Fs, window=window, noverlap=NOverlap, 
                                            nfft=NFFT, nperseg=NFFT, mode='complex')     

    # Frequency shifting and representation of magnitude in dB
    STFTShift = np.fft.fftshift(STFT, axes=0)                # fftshift means 0 Doppler frequency is at center
    STFTShiftDB = 20 * np.log10(np.abs(STFTShift))           # Represent in dB
    STFTShiftDB = STFTShiftDB - np.max(STFTShiftDB)          # Normalize data to max value (i.e., the max value is 0dB)

    dAxis = np.fft.fftshift(dAxis)
    ind_min = np.argmax(dAxis > -2000)
    ind_max = np.argmax(dAxis > 2000)

    STFTFinal = STFTShiftDB[ind_min:ind_max]
    STFTFinal = np.where(STFTFinal < -40, -40, STFTFinal)
    STFTFinal = np.where(STFTFinal > -3, -3, STFTFinal)
    STFTFinal = np.repeat(STFTFinal, 10, axis=1)
    return STFTFinal


def get_mfcc(radarData):

    f0 = 60e9       	# radar operating frequency
    Fs = 2*f0/1e6       # sampling frequency

    radar1D = radarData.T.ravel()
    radar1D = radar1D - radar1D.mean()
    mfcc_final = mfcc(np.real(radar1D), sr=30, n_mfcc=180)
    # print(mfcc_final.shape, mfcc_final.max(), mfcc_final.min())
    # mfcc_final = mfcc_final[3:20, 0:120]
    mfcc_final = mfcc_final[::-1, :]
    mfcc_final = np.where(mfcc_final < -20, -20, mfcc_final)
    mfcc_final = np.where(mfcc_final > 100, 100, mfcc_final)
    # mfcc_final = np.repeat(mfcc_final, 6.5, axis=0)
    return mfcc_final


def get_batch(radarData, mode):
    Nsamples = radarData.shape[0]
    if mode == 'stft':
        return np.array([get_stft(radarData[i, :]) for i in range(Nsamples)])
    elif mode == 'mfcc':
        return np.array([get_mfcc(radarData[i, :]) for i in range(Nsamples)])


if __name__ == "__main__":
    from datetime import datetime
    import matplotlib.pyplot as plt

    import radar
    # X, Y, class_labels = radar.getTrainData(source_dir='2021_10_13_data')
    # radar.cache('save', X, Y, class_labels)
    X, Y, class_labels = radar.cache('load')
    print(X.shape, Y.shape, class_labels)

    now = datetime.now()
    # X_STFT = get_stft(X[0])
    X_STFT = get_mfcc(X[0])
    plt.plot()
    # plt.imshow(X_STFT, cmap='jet', vmin=-40, vmax=-3, aspect='auto', origin='lower')
    plt.imshow(X_STFT, cmap='jet', aspect='auto', origin='lower')
    plt.axis('off')
    print(datetime.now() - now)
    plt.show()

    # code in between
    # print(X_STFT.shape)
    # X_input = reshape_features(X_STFT, type='dl')
    # print(X_input.shape)
    # y_one_hot = one_hot_dl(Y)
    # print(y_one_hot.shape)