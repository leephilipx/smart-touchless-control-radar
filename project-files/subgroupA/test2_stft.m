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

%-------------------------------------------------------
% Time domain signal: s1
%-------------------------------------------------------
Np = NTS*Nframe;

s2=reshape(s1,1,Np);

s2=s2-mean(s2);

Tt = 30; % Np/Fs;
Fs = 2*f0/1e6;

TimeAxis = linspace(0,Tt,Np);
FreqAxis = linspace(-Fs/2,Fs/2,Np);

fh3=figure(3);
    xticks=[TimeAxis(1):5:TimeAxis(end)];
    s2max=max(abs(s2));
    x1=xticks(1); 
    x2=xticks(end);
subplot(2,1,1)
    plot(TimeAxis,real(s2)/s2max,'r-');
    axis([x1 x2 -1 1]);
    ylabel('Real','FontSize',10);
    title('Time domain response','FontSize',10)
    set(gca,'XTick',xticks,'YTick',[-1 0 1],'FontSize',10);
subplot(2,1,2)
    plot(TimeAxis,imag(s2)/s2max,'b-');
    axis([x1 x2 -1 1]);
    xlabel('Time (s)','FontSize',10);
    ylabel('Imag','FontSize',10);
    set(gca,'XTick',xticks,'YTick',[-1 0 1],'FontSize',10);
set(fh3,'Position',[400 450 400 300])
print -djpeg fig3_TimeResponse.jpg

%-------------------------------------------------------
% Frequency domain analysis: s2
%-------------------------------------------------------
s2fft=fftshift(fft(s2));       % center is 0 Doppler frequency
s2fftdb=20*log10(abs(s2fft));
s2fftdb=s2fftdb-max(s2fftdb);  % Normalize data (maximum value is 0 dB)

fh4=figure(4);
    f2xticks = [FreqAxis(1) 0 FreqAxis(end)]/1e3;
    f2yticks = [-120:30:0];
    plot(FreqAxis/1e3,s2fftdb,'m-');
    xlim([f2xticks(1) f2xticks(end)]);
    ylim([-120 0]);
    xlabel('Frquency (kHz)','FontSize',10);
    ylabel('Spectral power (dB)','FontSize',10);
    title('Frequency domain response','FontSize',10)
    set(gca,'XTick',f2xticks,'YTick',f2yticks,'FontSize',10);
set(fh4,'Position',[400 65 400 300])
print -djpeg fig4_FreqResponse.jpg

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

fh5=figure(5);
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
set(fh5,'Position',[200 250 400 300])
print -djpeg fig5_STFT.jpg

saveas(fh5, replace(filename, '.h5', '.jpg'));

%-------------------------------------------------------
% Zoom in
%-------------------------------------------------------

% timeWin=[5 10];   % (s)
% x1b=timeWin(1); x2b=timeWin(2);

% fh4=figure(4);
%     displayScale=[-70 -40];
%     imagesc(tAxis,dAxis/1e3,s3db,displayScale); 
%     colormap('jet'); set(colorbar,'FontSize',10);
%     xlim([x1b x2b]);
%     ylim([f3yticks(1) f3yticks(end)]);
%     xlabel('Time (s)','FontSize',10);
%     ylabel('Doppler (kHz)','FontSize',10);
%     title('Doppler response in a given window','FontSize',10)
%     set(gca,'YDir','normal','XTick',[x1b x2b],'YTick',f3yticks,'FontSize',10);
%     set(fh4,'Color',[1 1 1],'Position',[500 110 400 300])
% % print -djpeg fig4_MicroDopplerWin1.jpg

%-------------------------------------------------------
% Zoom in with different display scale
%-------------------------------------------------------

% timeWin=[25 30];   % (s)
% x1b=timeWin(1); x2b=timeWin(2);
% fh5=figure(5);
%     displayScale=[-70 -40];
%     imagesc(tAxis,dAxis/1e3,s3db,displayScale); 
%     colormap('jet'); set(colorbar,'FontSize',10);
%     xlim([x1b x2b]);
%     ylim([f3yticks(1) f3yticks(end)]);
%     xlabel('Time (s)','FontSize',10);
%     ylabel('Doppler (kHz)','FontSize',10);
%     title('Doppler response in a given window','FontSize',10)
%     set(gca,'YDir','normal','XTick',[x1b x2b],'YTick',f3yticks,'FontSize',10);
%     set(fh5,'Color',[1 1 1],'Position',[900 110 400 300])
% % print -djpeg fig5_MicroDopplerWin2.jpg

disp('Completed!')

%eof