clear all; close all; clc

f0=60e9;       	% radar operating frequency
c=3e8;        	% speed of light
lambda=c/f0;    % radar wavelength

Rmin   = 0.2;
Rmax   = 2.2;
Rstep  = 4.8400e-04;    % meter

FrameRate = 200;

filename = "xm112 (0.5m) sample.h5";
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

% fast time axis to obtain range
% Range Vs Frame

%===================
R1abs = abs(s1);

axisRange = linspace(Rmin, Rmax, Nrange);   % 1x2272
axisFrame = [1:Nframe];                     % 1xNframe
axisTime  = axisFrame/FrameRate;            % 1xNframe

[rPeak rIndex] = max(R1abs);
R1abs = R1abs/max(rPeak);
rPeak = axisRange(rIndex);
rMaxcm = rPeak(100)*100; rMaxcm=round(rMaxcm*10)/10;

Tmax = round(max(axisTime)*100)/100;

y3tick=[Rmin*100 rMaxcm Rmax*100];

R1db=20*log10(R1abs);

fh1=figure(1);  % Range vs frame
    surf(axisFrame, axisRange*100, R1abs); 
    colormap(jet)
    shading interp
    xlim([1 Nframe]); 
    ylim([Rmin Rmax]*100); 
    zlim([0 1])
    xlabel('Sweep','fontsize',12)
    ylabel('Range (cm)','fontsize',12)
    title('Output signal profile','fontsize',12)
    set(gca,'XTick',[1 200:200:Nframe],'YTick',y3tick,'ZTick',[0 1])
    view(45,60)
    set(fh1,'Position',[10 450 400 300])
print -djpeg fig1_RangeSweep3D.jpg;

fh2=figure(2); % Range vs Time
    imagesc(axisFrame, axisRange*100, R1abs);
    xlabel('Sweep','fontsize',12)
    ylabel('Range (cm)','fontsize',12);
    title('Output signal profile','fontsize',12)    
    colormap(jet); colorbar
    xlim([1 Nframe]);
    ylim([Rmin Rmax]*100)
    set(gca,'YDir','normal','XTick',[1 200:200:Nframe],'YTick',y3tick)
    set(fh2,'Position',[10 65 400 300])
print -djpeg fig2_RangeSweep2D.jpg

test2_stft

%eof
 