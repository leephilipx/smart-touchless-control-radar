# Smart Touchless Control Radar

<br>

## This project is in fulfillment of NTU-EEE's EE3080 Design & Innovation Project.
- Project Title: E047 Smart Touchless control with Millimeter-Wave Radar Sensor and Artificial Intelligence
- Supervisor: Prof Lu Yilong
- Laboratory: Media Technology Lab (Loc: S2.2-B4-02)
- Group Members: Philip, Yi Wen (A), Wai Yeong (A), Davis (B), Jun De (B)

<br>

## Useful Links
- https://blogs.ntu.edu.sg/ee3080-2122s1-e047/
- https://ts.ntu.edu.sg/sites/intranet/cs/eee/UGProg/FTUGProg/dip/Pages/dipIntro.aspx
- https://acconeer-python-exploration.readthedocs.io/en/latest/index.html

<br>

## Folder Structure
- Folders with the prefix `acconeer` are sample/setup references files.
- `admin-files` contains files relating to the management aspect of the project. Latest update can be found in ```admin-files/weekly_updates/E047_WeeklyUpdate6_20210920.pptx```
- `project-files` contains actual scripts used by the different subgroups or provided by our supervisor.

<br>

## Key Dates
- [x] Week 4: Deadline for *Project Charter*
- [ ] Week 13: Deadline for *Project Report*
- [ ] Week 13: Deadline for *Peer Review*
- [ ] Week 14: *DIP Competition*

<br>

## GUI for Gesture Extraction
 - `gui-dip-demo/gesture_extractor_gui.py` or ` is a Python script developed by Philip to automate the process of extracting gesture samples from a continous recording from the GUI tool provided by Acconeer. This solves the alignment issue faced when recording one sample at a time.
 - It then saves the samples in `.npz` format with the intention of a small file size. To read data from these new files, please use `np.load('filename-000.npz')['sample']`.
 - For ease of running the tool, an alternative Windows executable file `gui-dip-demo/gesture_extractor_gui.exe` can be used too.  
     
   <img src="./admin-files/weekly_updates/Week 7 Gesture Extractor GUI.png" height="250px"/>  

 - Sample commands:&nbsp;&nbsp;`python gesture_extractor_gui.py`
```
usage: gesture_extractor_gui.py [-h]

DIP E047 - GUI for Gesture Extraction

optional arguments:
  -h, --help            show this help message and exit
```

<br>

## GUI for Data Collection&nbsp;&nbsp;`needs fixing`
 - `gui-dip-demo/daq_menu_v1.py` is a Python script developed by Philip and Wai Yeong via *Tkinter* and *Acconeer SDK* to assist in automating the process of recording and saving radar data in IQ format.  
     
   <img src="./admin-files/weekly_updates/Week 7 Data Collection GUI.png" height="400px"/>  

 - Sample commands:&nbsp;&nbsp;`python daq_menu_v1.py`,&nbsp;&nbsp;`python daq_menu_v1.py -l`,&nbsp;&nbsp;`python daq_menu_v1.py -p COM3`, `python daq_menu_v1.py -c xb112_config.json`
```
usage: daq_menu_v1.py [-h] [-p P] [-l] [-c C]

DIP E047 - GUI for Data Collection

optional arguments:
  -h, --help            show this help message and exit
  -p P, --p P, -port P, --port P
                        Manually specify COM port for serial connection
  -l, --l, -list, --list
                        Lists available serial ports
  -c C, --c C, -config C, --config C
                        Manually specify config file path, accepts a json file
```

<br>

#### The *readme* will be updated more as the project advances.
