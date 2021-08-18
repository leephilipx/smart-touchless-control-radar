clear all; close all; clc

filename = "data1fb1.h5";
info = h5info(filename);

data = h5read(filename, "/data");
% disp("Data size =");
% disp(size(data));
% Dimensions (frame, sensor, depth) for Envelope, IQ, Power bins
%            (frame, sensor, sweep, depth) for Sparse

data_info = jsondecode(string(h5read(filename, "/data_info")));
first_data_info = data_info(1, 1)  % (frame, sensor);

rss_version = string(h5read(C:\Users\p1553835\OneDrive\Documents\GitHub\smart-touchless-control-radar\project files, "/rss_version"));
lib_version = string(h5read(C:\Users\p1553835\OneDrive\Documents\GitHub\smart-touchless-control-radar\project files, "/lib_version"));
timestamp   = string(h5read(C:\Users\p1553835\OneDrive\Documents\GitHub\smart-touchless-control-radar\project files, "/timestamp"));

%end of loadtestfile

s1r = squeeze(data.r);
s1i = squeeze(data.i);
s1  = s1r + j*s1i;

[NTS Nframe]=size(s1)
Nrange = NTS;

Rmin   = 0.2;
Rmax   = 0.8;
Rstep  = 4.8400e-04;    % meter

% fast time axis to obtain range
% Range Vs Frame

FrameRate = 200;

%===================
R1abs = abs(s1);

axisRange = linspace(Rmin, Rmax, Nrange);   % 1x2272
axisFrame = [1:Nframe];                     % 1xNframe
axisTime  = axisFrame/FrameRate;            % 1xNframe

[rPeak rIndex] = max(R1abs);
R1abs = R1abs/max(rPeak);
rPeak = axisRange(rIndex);
rMaxcm = rPeak(500)*100; rMaxcm=round(rMaxcm*10)/10;

Tmax = round(max(axisTime)*100)/100;

y3tick=[Rmin*100 rMaxcm Rmax*100];

%--------------------------------------------------------------
fh1=figure(1);  % Range vs frame
    surf(axisFrame, axisRange*100, R1abs); 
    colormap(jet)
    shading interp
    xlim([1 Nframe]); 
    ylim([Rmin Rmax]*100); 
    zlim([0 1])
    xlabel('Sweep','fontsize',12)
    ylabel('Distance (cm)','fontsize',12)
    title('Output signal profile','fontsize',12)
    set(gca,'XTick',[1 200:200:Nframe],'YTick',y3tick,'ZTick',[0 1])
    view(45,60)
set(fh1,'Position',[10 550 500 400])
print -djpeg fig1_surf.jpg;

%--------------------------------------------------------------
R1db=20*log10(R1abs);
fh2=figure(2); % Range vs Time
    imagesc(axisFrame, axisRange*100, R1db, [-20 0]);
    xlabel('Sweep','fontsize',12)
    ylabel('Range (cm)','fontsize',12);
    title('Output signal profile','fontsize',12)    
    colormap(jet); colorbar
    xlim([1 Nframe]);
    ylim([Rmin Rmax]*100)
    set(gca,'YDir','normal','XTick',[1 200:200:Nframe],'YTick',y3tick)
    set(fh2,'Position',[10 60 500 400])
print -djpeg fig2_RangeSweep.jpg

%eof
 