from tkinter import *
from tkinter import ttk
from daq_functions import AcconeerSensorDataCollection
from threading import Thread

class DataAcquisiton:

    def __init__(self, root, method, Nframes, config_path):
        self.radar = AcconeerSensorDataCollection(method=method, Nframes=Nframes, config_path=config_path)
        self.config_dict = self.radar.get_config_dict()
        self.sample_counter = 0
        root.title("Data Collection GUI - DIP E047")
        mainframe = ttk.Frame(root, padding="12 12 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.create_widgets(mainframe)
        self.init_states()
        self.autoconnect()

    def create_widgets(self, mainframe):
        # ---------
        self.connect_b = ttk.Button(mainframe, text=" Auto Connect (Serial Port) ", command=self.autoconnect)
        self.connect_b.grid(row=1, column=1, columnspan=2, sticky=(W))
        self.log_label = ttk.Label(mainframe, text="Please connect the sensor!", foreground="red")
        self.log_label.grid(row=1, column=3, columnspan=2, sticky=(E))
        # ---------
        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=2, column=1, columnspan=4, sticky=(E,W))
        self.start_b = ttk.Button(mainframe, text="Start", command=self.start)
        self.start_b.grid(row=3, column=1, sticky=(E,W))
        self.next_b = ttk.Button(mainframe, text="Next", command=self.next)
        self.next_b.grid(row=3, column=2, sticky=(E,W))
        self.discard_b = ttk.Button(mainframe, text="Discard", command=self.discard)
        self.discard_b.grid(row=3, column=3, sticky=(E,W))
        self.stop_b = ttk.Button(mainframe, text="Stop", command=self.stop)
        self.stop_b.grid(row=3, column=4, sticky=(E,W))
        self.filename_l = ttk.Label(mainframe, text="Filename (*.npy): ")
        self.filename_l.grid(row=4, column=1, sticky=(W))
        self.filename_val = StringVar()
        self.filename_val.set(".npy")
        self.filename_entry = ttk.Entry(mainframe, textvariable=self.filename_val)
        self.filename_entry.grid(row=4, column=2, columnspan=2, sticky=(E,W))
        self.counter_l = ttk.Label(mainframe, text="Count = 0", foreground="blue")
        self.counter_l.grid(row=4, column=4, sticky=(E))
        # ---------
        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=5, column=1, columnspan=4, sticky=(E,W))
        self.frameMagPlot = ttk.Labelframe(mainframe, text='Magnitude Plot', height=250, width=500)
        self.frameMagPlot.grid(row=6, column=1, columnspan=4, sticky=(E,W))
        # ---------
        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=9, column=1, columnspan=4, sticky=(E,W))
        self.frameConfigInfo = ttk.Labelframe(mainframe, text='Config Info', height=250, width=500)
        self.frameConfigInfo.grid(row=10, column=1, columnspan=4, sticky=(E,W))
        config_info = '\n'.join([f'{elem[0]}: {elem[1]}' for elem in self.config_dict.items()])
        self.config_l = ttk.Label(self.frameConfigInfo, text=config_info).grid(row=10, column=1, columnspan=4, sticky=(N,W))
        # ---------
        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
        self.filename_entry.focus()
        root.bind("<Return>", self.start)    

    def init_states(self):
        self.start_b['state'] = DISABLED
        self.next_b['state'] = DISABLED
        self.discard_b['state'] = DISABLED
        self.stop_b['state'] = DISABLED

    def autoconnect(self):
        if self.radar.connection_state:
            self.radar.disconnect_sensor()
        port = self.radar.autoconnect_serial_port()
        if port is not None:
            self.start_b['state'] = NORMAL
            self.log_label['text'] = f'Connection success on {port}!'
        else:
            self.start_b['state'] = DISABLED
            self.log_label['text'] = f'Connection failed!'

    def update_counter(self, increment=True):
        if increment:
            self.sample_counter += 1
        else:
            self.sample_counter -= 1
        self.counter_l['text'] = f'Count = {self.sample_counter}'

    def record(self):
        self.log_label['text'] = f'Recording hand gesture ...'
        self.update_counter()
        if self.radar.get_data() is not None:
            self.log_label['text'] = f'Hand gesture recorded!'
            self.next_b['state'] = NORMAL
            self.discard_b['state'] = NORMAL
            self.stop_b['state'] = NORMAL
        else:
            self.update_counter(increment=False)
            self.radar.stop_session()
            self.log_label['text'] = f'Data reading failed! Restart ...'
            self.start_b['state'] = NORMAL
            self.next_b['state'] = DISABLED
            self.discard_b['state'] = DISABLED
            self.stop_b['state'] = DISABLED

    def save_file(self):
        filename = self.filename_val.get()
        self.radar.save_data(npy_filename=f'{filename[:-4]}-{str(self.sample_counter).zfill(2)}.npy')

    def start(self):
        try:
            if self.filename_val.get() in ['*.npy', '.npy', ''] or self.filename_val.get()[-4:] != '.npy':
                self.log_label['text'] = f'Please enter a valid filename!'
                return
        except:
            self.log_label['text'] = f'Please enter a valid filename!'
            return
        self.filename_entry['state'] = DISABLED
        self.log_label['text'] = f'Starting a session ...'
        self.start_b['state'] = DISABLED
        self.next_b['state'] = DISABLED
        self.discard_b['state'] = DISABLED
        self.stop_b['state'] = DISABLED
        if self.radar.start_session():
            Thread(target=self.record).start()
        else:
            self.log_label['text'] = f'Failed to start session!'
            self.start_b['state'] = NORMAL
            
    def next(self):
        self.next_b['state'] = DISABLED
        self.discard_b['state'] = DISABLED
        self.stop_b['state'] = DISABLED
        Thread(target=self.record).start()
        self.save_file()

    def discard(self):
        self.update_counter(increment=False)
        self.discard_b['state'] = DISABLED
        self.log_label['text'] = f'Recording discarded!'

    def stop(self):
        self.save_file()
        self.radar.stop_session()
        self.start_b['state'] = NORMAL
        self.next_b['state'] = DISABLED
        self.discard_b['state'] = DISABLED
        self.stop_b['state'] = DISABLED
        self.log_label['text'] = f'Recording ended, start a new session!'


if __name__ == '__main__':
    root = Tk()
    DataAcquisiton(root, method='serial', Nframes=128, config_path='sensor_configs.json')
    root.mainloop()
