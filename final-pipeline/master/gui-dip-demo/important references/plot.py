import warnings

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

from acconeer.exptool import clients, configs, utils
from acconeer.exptool.pg_process import PGProccessDiedException, PGProcess
from acconeer.exptool.structs import configbase

from daq_functions import AcconeerSensorDataCollection
from iq import ProcessingConfig

radar = AcconeerSensorDataCollection(method='serial', Nframes=128, config_path='sensor_configs.json')
sensor_config = radar.get_config()
processing_config = radar.get_processing_config()
sensor_config.sensor = args.sensors

session_info = client.setup_session(sensor_config)
pg_updater = PGUpdater(sensor_config, processing_config, session_info)
pg_process = PGProcess(pg_updater)

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
        # num_sensors = len(self.sensor_config.sensor)

        # self.ampl_plot = win.addPlot(row=0, col=0, colspan=num_sensors)
        # self.ampl_plot.setMenuEnabled(False)
        # self.ampl_plot.setMouseEnabled(x=False, y=False)
        # self.ampl_plot.hideButtons()
        # self.ampl_plot.showGrid(x=True, y=True)
        # self.ampl_plot.setLabel("bottom", "Depth (m)")
        # self.ampl_plot.setLabel("left", "Amplitude")
        # self.ampl_plot.setXRange(*self.depths.take((0, -1)))
        # self.ampl_plot.addLegend(offset=(-10, 10))

        # self.phase_plot = win.addPlot(row=1, col=0, colspan=num_sensors)
        # self.phase_plot.setMenuEnabled(False)
        # self.phase_plot.setMouseEnabled(x=False, y=False)
        # self.phase_plot.hideButtons()
        # self.phase_plot.showGrid(x=True, y=True)
        # self.phase_plot.setLabel("bottom", "Depth (m)")
        # self.phase_plot.setLabel("left", "Phase")
        # self.phase_plot.setXRange(*self.depths.take((0, -1)))
        # self.phase_plot.setYRange(-np.pi, np.pi)
        # self.phase_plot.getAxis("left").setTicks(utils.pg_phase_ticks)

        self.ampl_curves = []
        self.phase_curves = []
        for i, sensor_id in enumerate(self.sensor_config.sensor):
            legend = "Sensor {}".format(sensor_id)
            pen = utils.pg_pen_cycler(i)
            ampl_curve = self.ampl_plot.plot(pen=pen, name=legend)
            phase_curve = self.phase_plot.plot(pen=pen)
            self.ampl_curves.append(ampl_curve)
            self.phase_curves.append(phase_curve)

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
        sweeps = d["data"]
        histories = np.abs(d["history"])

        ampls = np.abs(sweeps)

        for i, _ in enumerate(self.sensor_config.sensor):
            self.ampl_curves[i].setData(self.depths, ampls[i])
            self.phase_curves[i].setData(self.depths, np.angle(sweeps[i]))

            im = self.history_ims[i]
            history = histories[:, i]
            im.updateImage(history, levels=(0, 1.05 * history.max()))

        m = self.smooth_max.update(ampls)
        self.ampl_plot.setYRange(0, m)