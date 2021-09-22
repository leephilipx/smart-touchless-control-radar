import warnings

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

from acconeer.exptool import clients, configs, utils
from acconeer.exptool.pg_process import PGProccessDiedException, PGProcess
from acconeer.exptool.structs import configbase

from daq_functions import AcconeerSensorDataCollection


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
        
        win.setGeometry(0, 0, 1000, 500)
        win.setWindowTitle("Data Collection GUI - DIP E047 v3.0")
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

        btnStart = QtGui.QPushButton("Start", win)
        btnStart.clicked.connect(self.btn_start)
        btnStart.move(100,0)
        btnNext = QtGui.QPushButton("Next", win)
        btnNext.clicked.connect(self.btn_next)
        btnNext.move(200,0)
        btnDiscard = QtGui.QPushButton("Discard", win)
        btnDiscard.clicked.connect(self.btn_discard)
        btnDiscard.move(300,0)
        btnStop = QtGui.QPushButton("Stop", win)
        btnStop.clicked.connect(self.btn_stop)
        btnStop.move(400,0)
        self.txtFilename = QtGui.QPlainTextEdit("*.npy", win)
        self.txtFilename.move(500, 0)
        self.txtFilename.resize(150, 25)

    def btn_start(self):
        filename_string = self.txtFilename.toPlainText()
        try:
            if filename_string in ['*.npy', '.npy', ''] or filename_string[-4:] != '.npy':
                log = f'Please enter a valid filename!'
                print(log)
                return
        except:
            log = f'Please enter a valid filename!'
            print(log)

        # while not interrupt_handler.got_signal:
#     info, data = client.get_next()
#     plot_data = processor.process(data, info)

#     if plot_data is not None:
#         try:
#             pg_process.put_data(plot_data)
#         except PGProccessDiedException:
#             break


    def btn_next(self):
        pass
        
    def btn_discard(self):
        print('bunbun')

    def btn_stop(self):
        print('bunbun')
    
    def get_filename(self):
        print('bunbun')

    def update(self, d):
        histories = np.abs(d["history"])
        for i, _ in enumerate(self.sensor_config.sensor):
            im = self.history_ims[i]
            history = histories[:, i]
            im.updateImage(history, levels=(0, 1.05 * history.max()))


radar = AcconeerSensorDataCollection()

port = utils.autodetect_serial_port()
client = clients.UARTClient(port)
client.squeeze = False

processing_config = get_processing_config()

session_info = client.setup_session(radar.sensor_config)
pg_updater = PGUpdater(radar.sensor_config, processing_config, session_info)
pg_process = PGProcess(pg_updater)
pg_process.start()
client.start_session()

interrupt_handler = utils.ExampleInterruptHandler()

processor = Processor(radar.sensor_config, processing_config, session_info)

while True:
    pass


# print("Disconnecting...")
# pg_process.close()
# client.disconnect()
