from tkinter import *
from tkinter import ttk
from threading import Thread
import argparse
import h5py
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


class DataAcquisiton:

    def __init__(self, root, range=[30,60]):
        self.Nframes = 64
        self.rangeList = np.arange(range[0], range[1]+1, 10)
        self.frame_start = 1
        root.title("Gesture Splitter - DIP E047 v1.0")
        mainframe = ttk.Frame(root, padding="12 12 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.create_widgets(mainframe)
        self.init_states()
        self.plot_mag()

    def create_widgets(self, mainframe):
        # ---------
        self.filename_label = ttk.Label(mainframe, text="Filename (*.h5): ")
        self.filename_label.grid(row=1, column=1, columnspan=2, sticky=(W))
        self.filename_val = StringVar()
        self.filename_val.set(".h5")
        self.filename_entry = ttk.Entry(mainframe, textvariable=self.filename_val)
        self.filename_entry.grid(row=1, column=3, columnspan=3, sticky=(E,W))
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
        self.selected_frame_label = ttk.Button(mainframe, text="Extract: 1-64", command=lambda:print('bun'))
        self.selected_frame_label.grid(row=3, column=4, sticky=(E))
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
        self.fig = plt.figure(figsize=(10,2))
        self.ax = self.fig.add_subplot(111)
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frameMagPlot)
        self.canvas.get_tk_widget().grid(row=6, column=1, columnspan=8, sticky=(N,S,E,W))
        # ---------
        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
        self.filename_entry.focus()
        root.bind("<Return>", print)

    def init_states(self, *args):
        # self.start_b['state'] = DISABLED
        pass
    
    def shift_frame(self, amount):
        self.frame_start += amount
        self.plot_mag()

    def plot_mag(self, *args):
        self.plot_Nframes = self.Nframes * 2
        hf = h5py.File('gui-dip-demo/data1fb1.h5', 'r')
        data = np.abs(np.squeeze(np.array(hf['/data']))).T
        normalised_data = data / np.max(data)
        NTS = data.shape[0]
        
        hf.close()
        x_range_start = self.frame_start - 1
        x_range_end = x_range_start + self.plot_Nframes - 1
        self.ax.clear()
        self.ax.imshow(normalised_data[:, x_range_start:x_range_end+1], aspect='auto', origin='lower', cmap='jet')
        xticks = np.linspace(x_range_start, x_range_end, 9).astype(int)
        yticks = np.linspace(0, NTS-1, self.rangeList.shape[0])
        
        self.info_label['text'] = f'{normalised_data.shape} {normalised_data[:, x_range_start:x_range_end].shape} {[x_range_start, x_range_end]}'
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Range (cm)')
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xticks+1)
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(self.rangeList)
        self.ax.axvline(0, c='red')
        self.ax.axvline(0+self.Nframes-1, c='red')
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
