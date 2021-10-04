from tkinter import *
from tkinter import ttk, filedialog, simpledialog
from argparse import ArgumentParser
from h5py import File
from json import loads
from os import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DataAcquisiton:

    def __init__(self, root):
        self.root = root
        self.Nframes = 64
        self.extracted_count = 0
        self.extracted_samples = []
        self.root.title("Gesture Extractor GUI - DIP E047 v1.3.2")
        # self.root.minsize(920, 420)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.mainframe = ttk.Frame(self.root, padding="12 12 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.mainframe.pack(expand=True, fill='both', anchor='center')
        self.create_widgets()
        self.init_states()
        self.get_file()

    def create_widgets(self):
        # ---------
        self.file_button = ttk.Button(self.mainframe, text="Load h5 file", command=self.get_file)
        self.file_button.grid(row=1, column=1, sticky=(E,W))
        self.info_label = ttk.Label(self.mainframe, text="  Please load a h5 file!", foreground="red")
        self.info_label.grid(row=1, column=2, columnspan=2, sticky=(E,W))
        self.extract_button = ttk.Button(self.mainframe, text="Extract sample", command=self.extract_sample)
        self.extract_button.grid(row=1, column=4, columnspan=1, sticky=(E,W))
        self.save_button = ttk.Button(self.mainframe, text="Save all", command=self.save_samples)
        self.save_button.grid(row=1, column=5, columnspan=1, sticky=(E,W))
        self.Nframes_label = ttk.Label(self.mainframe, text="Frame size:", foreground='blue')
        self.Nframes_label.grid(row=1, column=6, sticky=(W))
        self.Nframes_val = IntVar()
        self.Nframes_val.set(self.Nframes)
        self.Nframes_entry = ttk.Entry(self.mainframe, textvariable=self.Nframes_val, width=5)
        self.Nframes_entry.grid(row=1, column=6, sticky=(E))
        # ---------
        ttk.Separator(self.mainframe, orient=HORIZONTAL).grid(row=2, column=1, columnspan=8, sticky=(E,W))
        self.decrement_N_button = ttk.Button(self.mainframe, text=f"- {self.Nframes_val.get()}", command=lambda:self.shift_frame(-self.Nframes_val.get()))
        self.decrement_N_button.grid(row=3, column=1, sticky=(E,W))
        self.decrement_five_button = ttk.Button(self.mainframe, text="- 5", command=lambda:self.shift_frame(-5))
        self.decrement_five_button.grid(row=3, column=2, sticky=(E,W))
        self.decrement_one_button = ttk.Button(self.mainframe, text="- 1", command=lambda:self.shift_frame(-1))
        self.decrement_one_button.grid(row=3, column=3, sticky=(E,W))
        self.increment_one_button = ttk.Button(self.mainframe, text="+ 1", command=lambda:self.shift_frame(1))
        self.increment_one_button.grid(row=3, column=4, sticky=(E,W))
        self.increment_five_button = ttk.Button(self.mainframe, text="+ 5", command=lambda:self.shift_frame(5))
        self.increment_five_button.grid(row=3, column=5, sticky=(E,W))
        self.increment_N_button = ttk.Button(self.mainframe, text=f"+ {self.Nframes_val.get()}", command=lambda:self.shift_frame(self.Nframes_val.get()))
        self.increment_N_button.grid(row=3, column=6, sticky=(E,W))
        # ---------
        ttk.Separator(self.mainframe, orient=HORIZONTAL).grid(row=4, column=1, columnspan=6, sticky=(E,W))
        self.frameMagPlot = ttk.Labelframe(self.mainframe, text='Magnitude Plot')
        self.frameMagPlot.grid(row=5, column=1, columnspan=6, sticky=(N,S,E,W))
        self.fig = plt.figure(figsize=(7, 2))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frameMagPlot)
        self.canvas.get_tk_widget().grid(row=6, column=1, columnspan=6, sticky=(N,S,E,W))
        # ---------
        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
        root.bind("<Key>", self.update_button)
        self.Nframes_entry.focus()

    def init_states(self, *args):
        self.increment_one_button['state'] = DISABLED
        self.increment_five_button['state'] = DISABLED
        self.increment_N_button['state'] = DISABLED
        self.decrement_one_button['state'] = DISABLED
        self.decrement_five_button['state'] = DISABLED
        self.decrement_N_button['state'] = DISABLED
        self.Nframes_entry['state'] = DISABLED
        self.extract_button['state'] = DISABLED
        self.save_button['state'] = DISABLED
    
    def update_button(self, *args):
        try:
            self.Nframes = self.Nframes_val.get()
            assert type(self.Nframes) == int
            assert self.Nframes > 0
        except:
            self.info_label['text'] = f'  Invalid frame size!'
            return
        self.info_label['text'] = f'  Frame size set to {self.Nframes}!'
        if self.Nframes > self.total_frames:
            self.Nframes = self.total_frames
            self.Nframes_val.set(self.total_frames)
            self.info_label['text'] = f'  Frame size capped at a max of {self.total_frames}!'
        self.increment_N_button['text'] = f'+ {self.Nframes}'
        self.decrement_N_button['text'] = f'- {self.Nframes}'
        self.read_file()
        self.plot_mag()

    def get_file(self, *args):
        self.extracted_samples = []
        self.extracted_count = 0
        self.ax.clear()
        self.canvas.draw_idle()
        self.init_states()
        self.root.update_idletasks()
        self.h5_path = filedialog.askopenfilename(filetypes=[('h5 file (*.h5)', '.h5')])
        if len(self.h5_path) and self.h5_path.endswith('.h5'):
            self.info_label['text'] = f'  {path.basename(self.h5_path)} loaded!'
            self.read_file()
        else:
            self.info_label['text'] = '  Unsuccessful, please try again!'

    def read_file(self, *args):
        self.frame_start = 0
        hf = File(self.h5_path, 'r')
        try:
            self.data = np.squeeze(np.array(hf['data']))
            try:
                range_interval = loads(str(np.squeeze(np.array(hf['sensor_config_dump'])))[2:-1])['range_interval']
            except:
                range_interval = loads(str(np.squeeze(np.array(hf['sensor_config_dump']))))['range_interval']
            self.range_interval = [int(100*item) for item in range_interval]
            self.total_frames, self.NTS = self.data.shape
            if self.total_frames < 32:
                self.info_label['text'] = f'  Insufficient frames! [Nframes={self.total_frames}]'
                return
            elif self.total_frames < 64:
                self.Nframes = 32
                self.Nframes_val.set(self.Nframes)
                self.increment_N_button['text'] = f'+ {self.Nframes}'
                self.decrement_N_button['text'] = f'- {self.Nframes}'
            self.increment_one_button['state'] = NORMAL
            self.increment_five_button['state'] = NORMAL
            self.increment_N_button['state'] = NORMAL
            self.decrement_one_button['state'] = NORMAL
            self.decrement_five_button['state'] = NORMAL
            self.decrement_N_button['state'] = NORMAL
            self.Nframes_entry['state'] = NORMAL
            self.extract_button['state'] = NORMAL
            self.save_button['state'] = NORMAL
            self.normalised_data = np.abs(self.data).T
            self.normalised_data = np.hstack([np.zeros((self.NTS, self.Nframes)), self.normalised_data/np.max(self.normalised_data), np.zeros((self.NTS, self.Nframes))])
            self.rangeList = np.linspace(self.range_interval[0], self.range_interval[1], 6).astype(int)
            self.yticks = np.linspace(0, self.NTS-1, 6)
            self.plot_mag()
        except:
            self.info_label['text'] = f'  Invalid h5 file, not loaded!'
        hf.close()

    def plot_mag(self, *args):
        self.plot_Nframes = 3 * self.Nframes
        self.xticks = np.linspace(0, self.plot_Nframes-1, 4).astype(int)
        self.ax.clear()
        self.ax.imshow(self.normalised_data[:, self.frame_start:self.frame_start+self.plot_Nframes], aspect='auto', origin='lower', vmin=0, cmap='jet')
        self.ax.axvline(self.Nframes, c='red')
        self.ax.axvline(2*self.Nframes-1, c='red')
        self.ax.set_title(f' ')
        self.ax.set_xlabel(f'Frame {self.frame_start+1} to {self.frame_start+self.Nframes}')
        self.ax.set_ylabel('Range (cm)')
        self.ax.set_xticks(self.xticks)
        self.ax.set_xticklabels(self.xticks+self.frame_start-self.Nframes+1)
        self.ax.set_yticks(self.yticks)
        self.ax.set_yticklabels(self.rangeList)
        plt.tight_layout()
        self.canvas.draw_idle()
    
    def shift_frame(self, amount):
        self.frame_start += amount
        if self.frame_start < 0: self.frame_start = 0
        if self.frame_start >= (self.total_frames - self.Nframes): self.frame_start = self.total_frames - self.Nframes
        self.plot_mag()

    def extract_sample(self, *args):
        try:
            self.extracted_samples.append(self.data[self.frame_start:self.frame_start+self.Nframes, :])
            self.extracted_count += 1
            self.info_label['text'] = f'  Sample #{self.extracted_count} extracted!'
        except:
            self.info_label['text'] = f'  An error has occured!'

    def save_samples(self, *args):
        if self.extracted_count:
            self.info_label['text'] = '  Please select a directory!'
            self.root.update_idletasks()
            save_dir = filedialog.askdirectory()
            self.info_label['text'] = '  '
            if len(save_dir):
                self.root.update_idletasks()
                filename = simpledialog.askstring(title='Filename (no extension)', prompt='Please enter a filename to save your extracted samples!')
                if filename is None: return
                if len(filename):
                    filename = filename.lower()
                    ind = filename.find('.')
                    if ind != -1: filename = filename[:ind]
                    try:
                        for cnt in range(self.extracted_count):
                            np.savez_compressed(path.join(save_dir, f'{filename}-{str(cnt).zfill(3)}.npz'), sample=self.extracted_samples[cnt])
                        self.info_label['text'] = f"  {self.extracted_count} sample{'' if self.extracted_count == 1 else 's'} saved to {path.basename(save_dir)}!"
                    except:
                        self.info_label['text'] = '  An error has occured!'
                else:
                    self.info_label['text'] = '  Invalid filename!'
        else:
            self.info_label['text'] = '  Please extract samples first!'


if __name__ == '__main__':
    desc = 'DIP E047 - GUI for Gesture Extraction \n Python script developed by Philip to automate the process of extracting gesture samples from a continous recording obtained from the GUI tool provided by Acconeer. This fixes the alignment issue faced when recording one sample at a time. | The tool saves extracted samples in .npz format with the intention of a small file size. To read data from these new files, please use np.load(\'filename-000.npz\')[\'sample\'], where filename is a placeholder. | For ease of running the tool, an alternative Windows executable file gui-dip-demo/gesture_extractor_gui.exe is available too.'
    parser = ArgumentParser(description=desc)
    args = parser.parse_args()
    root = Tk()
    DataAcquisiton(root)
    root.mainloop()
