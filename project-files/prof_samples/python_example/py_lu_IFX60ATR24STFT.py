# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:19:25 2021

@author: xuyi
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
from os.path import dirname, join as pjoin

#import math
#import math

f0 = 60e9          #radar carrier frequency
c = 3e8            #speed of light
lamda = c/f0       #radar wavelength


#T = 10            #period for processing

Ns = 64 # no of sample per chirp
Nc = 64 # no of chirps per frame
N = 20 # no of frame 
#Fs = Ns*Nc*Nf
Ntot = Ns*Nc*N

#nBits = 16
#nChannels = 2

#d1 = np.fromfile('IFXradardata/Distance2GoL_record_20210706-165720.raw.bin'
#                 , dtype=np.complex64)
#d1 = d1[-5*Fs:]
mat_fname = pjoin("./tests/data/", 'close1_1.mat')
mat = scipy.io.loadmat(mat_fname)
d1 = mat['d']

d1 = d1.transpose()
d1 = d1.flatten()
d1max = np.max(d1)

#---------------------------------
# Fig. 1: Time response
#---------------------------------

d1standized = d1/d1max

#N = len(d1)
#Tt = N/Fs

t = np.arange(0, 30, 30/Ntot)

'''
plt.figure(figsize=(12, 3))
plt.plot(t, np.real(d1standized), 'r--')
plt.axis([0, 30, -1, 1])
plt.ylabel('Real')
plt.xlabel('Time')
plt.suptitle('real part time response')
plt.savefig('Fig1_1_RealPartTimeResponse2.jpg')
plt.show()
'''

#-------------------------------
# Fig. 2: Frequency response
#-------------------------------


d2 = fft(d1)
d2 = np.fft.fftshift(d2)
d2dB = 20*np.log10(abs(d2))
d2dB = d2dB-np.max(d2dB)

x = np.arange(-1, 1, 2/Ntot)
'''
plt.figure(figsize=(12, 6))
plt.plot(x, d2dB, 'r--')
plt.axis([-1, 1, -80, 0])
plt.ylabel('Power in dB')
plt.xlabel('Frequency')
plt.suptitle('Frequency response')
plt.savefig('Fig2_Frequency2.jpg')
plt.show()
'''

#-----------------------------------
# Fig. 3: Short Time Fourier Transform
#-----------------------------------

Overlap = 0.9
NFFT = 2**14
Noverlap = NFFT*Overlap

Win = signal.windows.taylor(NFFT, nbar=10, sll=120)
#Win = signal.windows.chebwin(NFFT, 120)

freqArray, timeArray, Sxx = signal.spectrogram(d1, fs = Ns*Nc, 
                                                window=Win, 
                                                nperseg = NFFT,
                                                noverlap=Noverlap, 
                                                nfft=NFFT)
#d3=Sxx
d3 = np.fft.fftshift(Sxx, axes = 0)
freqArray = np.fft.fftshift(freqArray)

d3dB = 20*np.log10(abs(d3))
d3dB = d3dB-np.max(np.max(d3dB))


plt.figure(figsize=(3, 3))
plt.pcolormesh(timeArray, freqArray/1e3, 
               d3dB, cmap='jet', vmin =-75, vmax = -45)
plt.axis([0, 15, 0.05, 0.08])
plt.axis('off')
# plt.savefig('Fig3_STFT3.jpg')
plt.show()


#========================================
#interpolation
#========================================
from scipy.interpolate import interp2d

f = interp2d(timeArray, freqArray/1e3, d3dB, kind='cubic')
xnew = np.arange(0, 30, 30/2**10)
ynew = np.arange(0, 2.048, 2/2**14)
data1 = f(xnew,ynew)
Xn, Yn = np.meshgrid(xnew, ynew)
plt.figure(figsize=(6, 6))

plt.pcolormesh(Xn, Yn, data1, cmap='jet', vmin =-100, vmax = -50)

plt.axis([0, 30, 0.055, 0.075])
plt.axis('off')
plt.savefig('Fig4_STFT.jpg')
plt.show()

