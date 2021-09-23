from tkinter import *
from tkinter import ttk, filedialog
from threading import Thread
import argparse
import h5py
import numpy as np
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


class DataAcquisiton:

    def __init__(self, root, range=[30,60]):
        self.Nframes = 64
        self.rangeList = np.arange(range[0], range[1]+1, 10)
        self.frame_start = 1
        self.frame_offset = 0
        root.title("Gesture Splitter - DIP E047 v1.0")
        mainframe = ttk.Frame(root, padding="12 12 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.create_widgets(mainframe)
        self.init_states()
        self.read_file()
        self.plot_mag()

    def create_widgets(self, mainframe):
        # ---------
        self.file_button = ttk.Button(mainframe, text="Load h5 file", command=self.get_file)
        self.file_button.grid(row=1, column=1, sticky=(E,W))
        self.selected_frame_label = ttk.Button(mainframe, text="Extract: 1-64", command=lambda:print('bun'))
        self.selected_frame_label.grid(row=1, column=2, columnspan=2, sticky=(E,W))
        self.Nframes_label = ttk.Label(mainframe, text="Frame size: ")
        self.Nframes_label.grid(row=1, column=7, sticky=(E))
        self.Nframes_val = StringVar()
        self.Nframes_val.set(self.Nframes)
        self.Nframes_entry = ttk.Entry(mainframe, textvariable=self.Nframes_val)
        self.Nframes_entry.grid(row=1, column=8, sticky=(E,W))
        # ---------
        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=2, column=1, columnspan=8, sticky=(E,W))
        self.decrement_twenty_button = ttk.Button(mainframe, text="- 20", command=lambda:self.shift_frame(-20))
        self.decrement_twenty_button.grid(row=3, column=1, sticky=(E,W))
        self.decrement_five_button = ttk.Button(mainframe, text="- 5", command=lambda:self.shift_frame(-5))
        self.decrement_five_button.grid(row=3, column=2, sticky=(E,W))
        self.decrement_one_button = ttk.Button(mainframe, text="- 1", command=lambda:self.shift_frame(-1))
        self.decrement_one_button.grid(row=3, column=3, sticky=(E,W))
        
        self.increment_one_button = ttk.Button(mainframe, text="+ 1", command=lambda:self.shift_frame(1))
        self.increment_one_button.grid(row=3, column=6, sticky=(E,W))
        self.increment_five_button = ttk.Button(mainframe, text="+ 5", command=lambda:self.shift_frame(5))
        self.increment_five_button.grid(row=3, column=7, sticky=(E,W))
        self.increment_twenty_button = ttk.Button(mainframe, text="+ 20", command=lambda:self.shift_frame(20))
        self.increment_twenty_button.grid(row=3, column=8, sticky=(E,W))
        self.info_label = ttk.Label(mainframe, text="Information Text", foreground="red")
        self.info_label.grid(row=4, column=1, columnspan=8, sticky=(E,W))
        # ---------
        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=5, column=1, columnspan=8, sticky=(E,W))
        self.frameMagPlot = ttk.Labelframe(mainframe, text='Magnitude Plot')
        self.frameMagPlot.grid(row=6, column=1, columnspan=8, sticky=(N,S,E,W))
        self.fig = plt.figure(figsize=(9, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frameMagPlot)
        self.canvas.get_tk_widget().grid(row=6, column=1, columnspan=8, sticky=(N,S,E,W))
        # ---------
        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
        root.bind("<Return>", print)

    def init_states(self, *args):
        # self.start_b['state'] = DISABLED
        pass
    
    def shift_frame(self, amount):
        self.frame_start += amount
        if self.frame_start < 1:
            self.frame_offset = 0
            self.frame_start = 1
        elif self.frame_start > (self.total_frames - self.Nframes):
            self.frame_offset = self.Nframes
            self.frame_start = self.total_frames - self.Nframes 
        elif self.frame_start > (self.total_frames - self.plot_Nframes):
            self.frame_offset += self.frame_start - (self.total_frames - self.plot_Nframes)
            self.frame_start = self.total_frames - self.plot_Nframes
        else:
            self.frame_offset = 0
        self.plot_mag()

    def get_file(self, *args):
        self.h5_path = filedialog.askopenfilename(filetypes=[('h5 file (*.h5)', '.h5')])
        print(self.h5_path)

    def read_file(self, *args):
        self.plot_Nframes = self.Nframes * 2
        hf = h5py.File('gui-dip-demo/data1fb1.h5', 'r')
        self.data = np.abs(np.squeeze(np.array(hf['/data']))).T
        self.normalised_data = self.data / np.max(self.data)
        self.NTS, self.total_frames = self.data.shape
        hf.close()

    def plot_mag(self, *args):
        
        x_range_start = self.frame_start - 1
        x_range_end = x_range_start + self.plot_Nframes - 1
        self.ax.clear()
        self.ax.imshow(self.normalised_data[:, x_range_start:x_range_end+1], aspect='auto', origin='lower', cmap='jet')
        xticks = np.linspace(0, self.plot_Nframes-1, 9).astype(int)
        yticks = np.linspace(0, self.NTS-1, self.rangeList.shape[0])
        
        self.info_label['text'] = f'{self.normalised_data.shape} {self.normalised_data[:, x_range_start:x_range_end].shape}'
        self.selected_frame_label['text'] = f'Extract Frames [{self.frame_start}, {self.frame_start+self.Nframes-1}]'
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Range (cm)')
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xticks+self.frame_start)
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(self.rangeList)
        self.ax.axvline(0+self.frame_offset, c='red')
        self.ax.axvline(0+self.frame_offset+self.Nframes-1, c='red')
        plt.tight_layout()
        self.canvas.draw_idle()
        


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='DIP E047 - GUI for Data Collection')
    # parser.add_argument('-p', '--p', '-port', '--port', type=str, help='Manually specify COM port for serial connection')
    # parser.add_argument('-l', '--l', '-list', '--list', action='store_true', help='Lists available serial ports')
    # parser.add_argument('-c', '--c', '-config', '--config', type=str, help='Manually specify config file path, accepts a json file')
    # args = parser.parse_args()

    # if args.l:
    #     print('>> Avaliable ports:', AcconeerSensorDataCollection(method='serial', Nframes=128).list_serial_ports())
    #     from sys import exit
    #     exit()

    # config_path = 'sensor_configs.json'
    # if args.c is not None: config_path = args.c

    root = Tk()
    DataAcquisiton(root, range=[30,60])
    root.mainloop()
