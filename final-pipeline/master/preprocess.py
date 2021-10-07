import numpy as np
import h5py, json
import matplotlib.pyplot as plt
from scipy import signal


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


def get_stft(data):

    graph_verbosity = True    # If set to true, graphs will be plotted

    f0 = 60e9       	# radar operating frequency
    c = 3e8             # speed of light
    wavelength = c/f0   # radar wavelength

    Tt = 30             # Np/Fs, total time taken for measurement
    Fs = 2*f0/1e6       # sampling frequency


    input_file = r'final-pipeline/master/models/push01.h5'
    hf = h5py.File(input_file, 'r')

    try:
        range_interval = json.loads(str(np.squeeze(np.array(hf['sensor_config_dump'])))[2:-1])['range_interval']
    except:
        range_interval = json.loads(str(np.squeeze(np.array(hf['sensor_config_dump']))))['range_interval']
    range_interval = [int(100*item) for item in range_interval]

    rangeList = np.linspace(range_interval[0], range_interval[1], 6).astype(int)
    radarData = np.squeeze(np.array(hf['data']))
    Nframe, NTS = radarData.shape

    hf.close()

    magnitudeData = np.abs(radarData).T
    normalizedMagnitudeData = magnitudeData / np.max(magnitudeData)

    # Predefine x and y ticks beforehand
    xticks = np.arange(0, Nframe, Nframe//10)
    yticks = np.linspace(0, NTS-1, rangeList.shape[0])

    # Number of data points
    Np = NTS * Nframe

    # Unwrap and normalize data
    radar1D = radarData.reshape(Np, 1).squeeze()
    radar1D = radar1D - radar1D.mean()
    radar1DNormalised = radar1D / np.max(np.abs(radar1D))

    # Some plotting parameters
    timeAxis = np.linspace(0, Tt, Np)


    # Some parameters to tweak
    overlapPercent = 0.8                                     # Percentage of data overlapping STFT
    NFFT = 1024 * 2   	                                     # FFT point numbers: 2^10=1024 2^12=4096 2^13=8192

    # Resolution parameters
    timeSegment = NFFT / Fs                                  # Time for each time segment in FFT processing
    dResolution = 1 / timeSegment                            # Doppler resolution (Hz)
    tResolution = (NFFT*(1-overlapPercent)) * 1/Fs           # Time resolution (s)
    vResolution = dResolution * wavelength / 2               # Velocity resolution (m/s)

    # Doppler window for FFT sidelobe suppression and STFT
    NOverlap = round(NFFT*overlapPercent);                   # Number of overlapping points
    window = signal.windows.taylor(NFFT, nbar=10, sll=80, norm=False)
    dAxis, tAxis, STFT = signal.spectrogram(radar1D, axis=0, fs=Fs, window=window, noverlap=NOverlap, nfft=NFFT,nperseg=NFFT)     

    # Frequency shifting and representation of magnitude in dB
    STFTShift = np.fft.fftshift(STFT, axes=0)                # fftshift means 0 Doppler frequency is at center
    # dAxis = dAxis - Fs/2;                                    # Shift center as 0 Doppler frequency
    dAxis = np.fft.fftshift(dAxis)                              # Shift center as 0 Doppler frequency
    STFTShiftDB = 20 * np.log10(np.abs(STFTShift))           # Represent in dB
    STFTShiftDB = STFTShiftDB - np.max(STFTShiftDB)          # Normalize data to max value (i.e., the max value is 0dB)

    # Show the STFT signature
    plt.figure(figsize=(8, 6));
    # plt.imshow(STFTShiftDB, aspect='auto', origin='lower', cmap='jet');
    plt.pcolormesh(tAxis, dAxis/1e3, STFTShiftDB, cmap='jet', vmin =-40, vmax = -3)
    plt.colorbar();
    plt.title('Doppler-Time Response (STFT)');
    plt.xlabel('Time (s)'); plt.ylabel('Doppler (kHz)');
    plt.xlim((0,0.5));
    plt.ylim((-2,2));
    plt.show();