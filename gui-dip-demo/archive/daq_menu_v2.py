from tkinter import *
from tkinter import ttk
from daq_functions import AcconeerSensorDataCollection
from threading import Thread
from multiprocessing import Process
import warnings
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from acconeer.exptool import clients, utils
from acconeer.exptool.pg_process import PGProccessDiedException, PGProcess
from acconeer.exptool.structs import configbase
import os


class DataAcquisiton:

    def __init__(self, root, method, Nframes, config_path):
        self.Nframes = Nframes
        self.config_path = config_path
        self.sample_counter = 0
        root.title("Data Collection GUI - DIP E047 v2.0")
        mainframe = ttk.Frame(root, padding="12 12 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.create_widgets(mainframe)
        self.init_states()

    def create_widgets(self, mainframe):
        # ---------
        self.load_config_b = ttk.Button(mainframe, text="Load Configs", command=self.loadconfigs)
        self.load_config_b.grid(row=1, column=1, sticky=(E,W))
        self.log_label = ttk.Label(mainframe, text="", foreground="red")
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
        self.filename_val.set("gg.npy")
        self.filename_entry = ttk.Entry(mainframe, textvariable=self.filename_val)
        self.filename_entry.grid(row=4, column=2, columnspan=2, sticky=(E,W))
        self.counter_l = ttk.Label(mainframe, text="Count = 0", foreground="blue")
        self.counter_l.grid(row=4, column=4, sticky=(E))
        # ---------
        ttk.Separator(mainframe, orient=HORIZONTAL).grid(row=9, column=1, columnspan=4, sticky=(E,W))
        self.frameConfigInfo = ttk.Labelframe(mainframe, text='Config Info', height=250, width=500)
        self.frameConfigInfo.grid(row=10, column=1, columnspan=4, sticky=(E,W))
        self.config_l = ttk.Label(self.frameConfigInfo, text="Please load the config file!")
        self.config_l.grid(row=10, column=1, columnspan=4, sticky=(N,W))
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

    def loadconfigs(self):
        self.radar = AcconeerSensorDataCollection(config_path=self.config_path)
        self.config_dict = self.radar.get_config_dict()
        config_info = '\n'.join([f'{elem[0]}: {elem[1]}' for elem in self.config_dict.items()])
        self.config_l['text'] = config_info
        self.start_b['state'] = NORMAL
        self.log_label['text'] = f'Config file loaded!'

    def update_counter(self, increment=True):
        if increment:
            self.sample_counter += 1
        else:
            self.sample_counter -= 1
        self.counter_l['text'] = f'Count = {self.sample_counter}'

    def record(self):
        # try:
        self.log_label['text'] = f'Recording hand gesture ...'
        self.update_counter()
        os.environ['dip_iq_window_param'] = 'window_open'
        Process(target=iq_main, args=(self.radar.sensor_config, self.Nframes)).start()
        while os.environ['dip_iq_window_param'] == 'window_open':
            pass
        # except:
        #     self.update_counter(increment=False)
        #     self.log_label['text'] = f'Data reading failed! Restart ...'
        #     self.start_b['state'] = NORMAL
        #     self.next_b['state'] = DISABLED
        #     self.discard_b['state'] = DISABLED
        #     self.stop_b['state'] = DISABLED

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
        self.load_config_b['state'] = DISABLED
        self.record()
            
    def next(self):
        self.save_file()
        self.next_b['state'] = DISABLED
        self.discard_b['state'] = DISABLED
        self.stop_b['state'] = DISABLED
        self.record()

    def discard(self):
        self.update_counter(increment=False)
        self.discard_b['state'] = DISABLED
        self.log_label['text'] = f'Recording discarded!'

    def stop(self):
        self.save_file()
        self.start_b['state'] = NORMAL
        self.next_b['state'] = DISABLED
        self.discard_b['state'] = DISABLED
        self.stop_b['state'] = DISABLED
        self.load_config_b['state'] = NORMAL
        self.log_label['text'] = f'Recording ended!'

    def recorded(self):
        self.log_label['text'] = f'Hand gesture recorded!'
        self.next_b['state'] = NORMAL
        self.discard_b['state'] = NORMAL
        self.stop_b['state'] = NORMAL

## Modified from iq.py ##

def iq_main(sensor_config, Nframes):

    port = utils.autodetect_serial_port()
    client = clients.UARTClient(port)
    client.squeeze = False

    processing_config = get_processing_config()
    
    session_info = client.setup_session(sensor_config)
    pg_updater = PGUpdater(sensor_config, processing_config, session_info)
    pg_process = PGProcess(pg_updater)
    pg_process.start()
    client.start_session()

    interrupt_handler = utils.ExampleInterruptHandler()

    processor = Processor(sensor_config, processing_config, session_info)

    while not interrupt_handler.got_signal:
        for i in range(Nframes):
            info, data = client.get_next()
            plot_data = processor.process(data, info)

            if plot_data is not None:
                try:
                    pg_process.put_data(plot_data)
                except PGProccessDiedException:
                    pg_process.close()
                    client.disconnect()
                    raise AssertionError
                    break
        break
    
    os.environ['dip_iq_window_param'] = 'window_close'
    pg_process.close()
    client.disconnect()
    

class ProcessingConfig(configbase.ProcessingConfig):
    VERSION = 2

    history_length = configbase.IntParameter(
        default_value=100,
        limits=(10, 1000),
        label="History length",
        order=0,
    )

    sf = configbase.FloatParameter(
        label="Smoothing factor",
        default_value=None,
        limits=(0.1, 0.999),
        decimals=3,
        optional=True,
        optional_label="Enable filter",
        optional_default_set_value=0.9,
        updateable=True,
        order=10,
    )


get_processing_config = ProcessingConfig


class Processor:
    def __init__(self, sensor_config, processing_config, session_info):
        depths = utils.get_range_depths(sensor_config, session_info)
        num_depths = depths.size
        num_sensors = len(sensor_config.sensor)
        history_length = processing_config.history_length
        self.history = np.zeros([history_length, num_sensors, num_depths], dtype="complex")
        self.lp_data = np.zeros([num_sensors, num_depths], dtype="complex")
        self.update_index = 0
        self.update_processing_config(processing_config)

    def update_processing_config(self, processing_config):
        self.sf = processing_config.sf if processing_config.sf is not None else 0.0

    def dynamic_sf(self, static_sf):
        return min(static_sf, 1.0 - 1.0 / (1.0 + self.update_index))

    def process(self, data, data_info=None):
        if data_info is None:
            warnings.warn(
                "To leave out data_info or set to None is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )

        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = data

        sf = self.dynamic_sf(self.sf)
        self.lp_data = sf * self.lp_data + (1 - sf) * data

        self.update_index += 1

        return {
            "data": self.lp_data,
            "history": self.history,
        }


class PGUpdater:
    def __init__(self, sensor_config, processing_config, session_info):
        self.sensor_config = sensor_config
        self.processing_config = processing_config

        self.depths = utils.get_range_depths(sensor_config, session_info)
        self.depth_res = session_info["step_length_m"]
        self.smooth_max = utils.SmoothMax(sensor_config.update_rate)

    def setup(self, win):

        rate = self.sensor_config.update_rate
        xlabel = "Sweeps" if rate is None else "Time (s)"
        x_scale = 1.0 if rate is None else 1.0 / rate
        y_scale = self.depth_res
        x_offset = -self.processing_config.history_length * x_scale
        y_offset = self.depths[0] - 0.5 * self.depth_res
        is_single_sensor = len(self.sensor_config.sensor) == 1

        self.history_plots = []
        self.history_ims = []
        for i, sensor_id in enumerate(self.sensor_config.sensor):
            title = None if is_single_sensor else "Sensor {}".format(sensor_id)
            plot = win.addPlot(row=2, col=i, title=title)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.setLabel("bottom", xlabel)
            plot.setLabel("left", "Depth (m)")
            im = pg.ImageItem(autoDownsample=True)
            im.setLookupTable(utils.pg_mpl_cmap("viridis"))
            im.resetTransform()
            tr = QtGui.QTransform()
            tr.translate(x_offset, y_offset)
            tr.scale(x_scale, y_scale)
            im.setTransform(tr)
            plot.addItem(im)
            self.history_plots.append(plot)
            self.history_ims.append(im)

    def update(self, d):
        histories = np.abs(d["history"])
        for i, _ in enumerate(self.sensor_config.sensor):
            im = self.history_ims[i]
            history = histories[:, i]
            im.updateImage(history, levels=(0, 1.05 * history.max()))



if __name__ == "__main__":
    root = Tk()
    daq = DataAcquisiton(root, method='serial', Nframes=128, config_path='sensor_configs.json')
    root.mainloop()
