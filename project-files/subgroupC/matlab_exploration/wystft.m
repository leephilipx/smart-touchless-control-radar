clear all; close all; clc

f0=60e9;       	% radar operating frequency
c=3e8;        	% speed of light
lambda=c/f0;    % radar wavelength

Rmin   = 0.2;
Rmax   = 2.2;
Rstep  = 4.8400e-04;    % meter

FrameRate = 200;

filename = "testfile1.h5";
info = h5info(filename);

data = h5read(filename, "/data");
% disp("Data size =");
% disp(size(data));
% Dimensions (frame, sensor, depth) for Envelope, IQ, Power bins
%            (frame, sensor, sweep, depth) for Sparse

data_info = jsondecode(string(h5read(filename, "/data_info")));
first_data_info = data_info(1, 1)  % (frame, sensor);

% rss_version = string(h5read(filename, "/rss_version"))
% lib_version = string(h5read(filename, "/lib_version"))
% timestamp   = string(h5read(filename, "/timestamp"))

%end of loadtestfile

s1r = squeeze(data.r);  % 2272x612  11123712  double
s1i = squeeze(data.i);  % 2272x612  11123712  double
s1  = s1r + j*s1i;      % 2272x612  22247424  double complex 

[NTS Nframe]=size(s1)   % NTS = 2272, Nframe = 612
Nrange = NTS;

R1abs = abs(s1);

axisRange = linspace(Rmin, Rmax, Nrange);   % 1x2272
axisFrame = [1:Nframe];                     % 1xNframe
axisTime  = axisFrame/FrameRate;            % 1xNframe

[rPeak rIndex] = max(R1abs);
R1abs = R1abs/max(rPeak);
rPeak = axisRange(rIndex);
rMaxcm = rPeak(64)*100; rMaxcm=round(rMaxcm*10)/10;

Tmax = round(max(axisTime)*100)/100;

y3tick=[Rmin*100 rMaxcm Rmax*100];

R1db=20*log10(R1abs);

fh1=figure(1); % Range vs Time
    imagesc(axisFrame, axisRange*100, R1abs);
    xlabel('Sweep','fontsize',12)
    ylabel('Range (cm)','fontsize',12);
    title('Output signal profile','fontsize',12)    
    colormap(jet); colorbar
    xlim([1 Nframe]);
    ylim([Rmin Rmax]*100)
    set(gca,'YDir','normal','XTick',[1 200:200:Nframe],'YTick',y3tick)
    set(fh1,'Position',[10 65 400 300])
print -djpeg fig1_RangeSweep2D.jpg

%-----------------------------------------------------------
% radar parameters required:
%-----------------------------------------------------------
% f0:           Carrier frequency (Hz)
% fs:           Sampling rate
% OVERLAP:      percentage of data overlapping in STFT (e.g., 0.9)
% NFFT:         FFT point number (e.g., 128, 256)
% s1:           Echo signal vector (complex)
% t:            Time axis vector
% T:            Total sampling time (sec)
%-----------------------------------------------------------

Np = NTS*Nframe;

s2=reshape(s1,1,Np);

s2=s2-mean(s2);

Tt = 30; % Np/Fs;
Fs = 2*f0/1e6;

%-------------------------------------------------------
% Doppler analysis
%-------------------------------------------------------
OVERLAP=0.8;    % percentage of data overlapping in STFT
NFFT=1024*2;   	% FFT point number 2^10=1024 2^12=4096 2^13=8192

timeSegment = NFFT/Fs;                 % time for each time segment in FFT processing
dResolution = 1/timeSegment;              % Doppler resolution (Hz)
tResolution = (NFFT*(1-OVERLAP))*1/Fs; % time resolution (s)
vResolution = dResolution*lambda/2;       % velocity resolution (m/s)

DopWin=window(@taylorwin,NFFT,10,-80);  	% Doppler window for FFT sidelobe suppression
Noverlap=round(NFFT*OVERLAP);               % Number of overlap points
[STFT,dAxis,tAxis]=spectrogram(s2,DopWin,Noverlap,NFFT,Fs);

s3=fftshift(STFT,1);    	% here fftshift means 0 Doppler frequency is at center
dAxis=dAxis-Fs/2;           % center is 0 Doppler frequency

s3db=20*log10(abs(s3));    	% in dB
s3db=s3db-max(max(s3db)); 	% Normalize data to maximum value (i.e., the maximum value is 0dB)

fh2=figure(2);
    f3xticks=[0:0.1:0.5];
    f3yticks=[-2.0 0 2.0];
    displayScale=[-18 -3];   % display scale in imagesc
    imagesc(tAxis,dAxis/1e3,s3db,displayScale); colormap('jet');
    xlim([f3xticks(1) f3xticks(end)]);
    ylim([f3yticks(1) f3yticks(end)]);
    xlabel('Time (s)','FontSize',10);
    ylabel('Doppler (kHz)','FontSize',10);
    title('Doppler-Time response','FontSize',10)
    colorbar; set(colorbar,'FontSize',10);
    set(gca,'YDir','normal','XTick',f3xticks,'YTick',f3yticks,'FontSize',10);
set(fh2,'Position',[200 250 400 300])
print -djpeg fig2_STFT.jpg
disp('Completed!')

%eof