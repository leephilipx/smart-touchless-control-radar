# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:05:28 2021

@author: xuyi
"""
import numpy as np
# import pywt
import matplotlib.pyplot as plt
import librosa
import librosa.display
#import math
#import math



f0 = 60e9          #radar carrier frequency
c = 3e8            #speed of light
lamda = c/f0       #radar wavelength


#T = 10            #period for processing

Ns = 64 # no of sample per chirp
Nc = 64 # no of chirps per frame
N = 15 # no of frame 
#Fs = Ns*Nc*Nf
Ntot = Ns*Nc*N


#nBits = 16
#nChannels = 2

#d1 = np.fromfile('IFXradardata/Distance2GoL_record_20210706-165720.raw.bin'
#                 , dtype=np.complex64)
#d1 = d1[-5*Fs:]
import scipy.io
mat = scipy.io.loadmat('Closeflattenned.mat')
d1 = mat['d']

d1 = d1.transpose()

d1 = d1.flatten()
d1 = d1-np.mean(d1)


mfccs = librosa.feature.mfcc(d1, sr = 1, n_mfcc = 80)

plt.figure(figsize = (4, 3))




#plt.imshow(abs(mfccs), cmap='jet', interpolation = 'bilinear', 
#           aspect = 'auto', vmin = -10, vmax = 25)

plt.imshow(abs(mfccs))
# plt.imshow(abs(mfccs), cmap='jet', interpolation = 'bilinear', 
#            aspect = 'auto', vmin = 0, vmax = 15)

plt.gca().invert_yaxis()
# plt.axis([0, 120, 3, 20])
plt.axis('off')
#plt.yticks(np.arange(1, 31))
#plt.xticks(0, len(d1))
plt.show()






