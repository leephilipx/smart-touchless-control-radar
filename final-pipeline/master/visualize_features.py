import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import librosa.display
from librosa.feature import mfcc
from multiprocessing import Process
import os
import io
import preprocess
from atpbar import atpbar, flush, register_reporter, find_reporter

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
    return dAxis, tAxis, STFTShiftDB

def get_stft_comparison_plot(X, Y, class_labels, index):
    fig, axes = plt.subplots(2,2)
    axes = axes.ravel()
    fig.suptitle('Doppler-Time Response (STFT)');
    for i, ax in zip(index, axes):
        x = X[i]
        y = Y[i]
        label = class_labels[y]
        dAxis, tAxis, STFT = get_stft(x)
        dAxis = np.fft.fftshift(dAxis)               # Shift center as 0 Doppler frequency
        ax.pcolormesh(tAxis, dAxis/1e3, STFT, cmap='jet', vmin =-40, vmax = -3)
        ax.set_title(f'{label} | {str(i).zfill(3)}');
        # ax.set_xlabel('Time (s)');
        ax.set_ylabel('Doppler (kHz)');
        ax.set_xlim((0,0.5));
        ax.set_ylim((-2,2));
    # plt.show();
    plt.savefig('stft.jpg')

def get_single_stft_plot(X, Y, class_labels, index, source_dir):
    x = X[index]
    y = Y[index]
    label = class_labels[y]
    dAxis, tAxis, STFT = get_stft(x)
    dAxis = np.fft.fftshift(dAxis)               # Shift center as 0 Doppler frequency
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            'project-files', 'radar_data', source_dir, 'images_stft')
    # fig = plt.figure()
    plt.pcolormesh(tAxis, dAxis/1e3, STFT, cmap='jet', vmin =-40, vmax = -3)
    plt.title(f'STFT: {label}-{str(index).zfill(3)}');
    plt.xlabel('Time (s)');
    plt.ylabel('Doppler (kHz)');
    # plt.xlim((0,0.5));
    plt.ylim((-2,2));
    # plt.show();
    plt.savefig(os.path.join(root_dir, f'{label}-{str(index).zfill(3)}.jpg'))

def get_mag_plot(X, Y, class_labels, index, source_dir):
    x = X[index]
    y = Y[index]
    label = class_labels[y]
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                'project-files', 'radar_data', source_dir, 'images_magnitude')
    plt.plot()
    plt.imshow(x, aspect='auto', origin='lower', cmap='jet');
    plt.title(f'Magnitude: {label}-{str(index).zfill(3)}');
    plt.xlabel('Frame'); 
    plt.ylabel('Range (cm)');
    plt.yticks(np.linspace(0, x.shape[0]-1, 5), labels=np.linspace(20, 60, 5).astype(int))
    # plt.show();
    plt.savefig(os.path.join(root_dir, f'{label}-{str(index).zfill(3)}.jpg'))

def get_mfcc_plot(index, class_labels, source_dir, X, Y):
    x = X[index]
    y = Y[index]
    label = class_labels[y]
    mfcc_final = preprocess.get_mfcc(x)
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            'project-files', 'radar_data', source_dir, 'images_mfcc')
    plt.plot()
    plt.imshow(mfcc_final, cmap='jet', aspect='auto')
    plt.title(f'MFCC: {label}-{str(index).zfill(3)}');
    plt.xlabel('Time'); 
    plt.ylabel('MFCC');
    plt.show();
    # plt.savefig(os.path.join(root_dir, f'{label}-{str(index).zfill(3)}.jpg'))

def multiproc_loop(mode, reporter, start, stop, class_labels, source_dir, X, Y):
    register_reporter(reporter)
    if mode == 'mag':
        X = preprocess.get_magnitude(X)
        for i in atpbar(range(start, stop), name=f'loop {start}-{stop-1}'):
            get_mag_plot(index=i, class_labels=class_labels, source_dir=source_dir, X=X, Y=Y)
    if mode == 'stft':
        for i in atpbar(range(start, stop), name=f'loop {start}-{stop-1}'):
            get_single_stft_plot(index=i, class_labels=class_labels, source_dir=source_dir, X=X, Y=Y)
    if mode == 'mfcc':
        for i in atpbar(range(start, stop), name=f'loop {start}-{stop-1}'):
            get_mfcc_plot(index=i, class_labels=class_labels, source_dir=source_dir, X=X, Y=Y)

def choose_plots(X, Y, class_labels, source_dir, multiproc, mode, number):
    if multiproc:                                     # Use multiprocessors
        reporter = find_reporter()
        processes = []
        for index in range(*number):
            p = Process(target=multiproc_loop, args=(mode, reporter, index, index+number[2], class_labels, 
                                                     source_dir, X, Y))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        flush()

    else:
        if mode == 'mag':                                   # Obtain individual mag plots
            X = preprocess.get_magnitude(X)
            for i in range(len(X)):
                get_mag_plot(index=i, class_labels=class_labels, source_dir=source_dir, X=X, Y=Y)
        
        if mode == 'stft':                                      # Obtain individual stft plots
            for i in number:
                get_single_stft_plot(index=i, class_labels=class_labels, source_dir=source_dir, X=X, Y=Y)

        if mode == 'stft_compare':                  # Compare stft of different gestures
            get_stft_comparison_plot(index=np.random.randint([0,250,500,750], [250,500,750,1000]), 
                                    class_labels=class_labels, X=X, Y=Y)
        if mode == 'mfcc':
            for i in number:
                get_mfcc_plot(index=i, class_labels=class_labels, source_dir=source_dir, X=X, Y=Y)

if __name__ == "__main__":
    import radar
    source_dir = '2021_10_20_data_new_gestures'
    X, Y, class_labels = radar.getTrainData(source_dir=source_dir)
    print(X.shape, Y.shape, class_labels)

    # Mode = 'mag'/'stft'/'stft_compare'/'mfcc'
    # choose_plots(X=X, Y=Y, class_labels=class_labels, source_dir=source_dir, multiproc=False, 
    #              mode='mfcc', number=[250,500,750,1000,1250])
    choose_plots(X=X, Y=Y, class_labels=class_labels, source_dir=source_dir, multiproc=True, 
                 mode='stft', number=[1000, 1300, 50])