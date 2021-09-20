import copy
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
import warnings
import webbrowser
from distutils.version import StrictVersion

import numpy as np
import pyqtgraph as pg

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QWidget,
)


HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.append(os.path.abspath(os.path.join(HERE, "ml")))
sys.path.append(os.path.abspath(os.path.join(HERE, "elements")))


try:
    from acconeer.exptool import SDK_VERSION, clients, configs, recording, utils
    from acconeer.exptool.structs import configbase

    import gui.data_processing
    from gui.elements.helper import (
        AdvancedSerialDialog,
        BiggerMessageBox,
        CollapsibleSection,
        Count,
        GUIArgumentParser,
        HandleAdvancedProcessData,
        Label,
        LoadState,
        SensorSelection,
        SessionInfoView,
        lib_version_up_to_date,
    )
    from gui.elements.modules import (
        MODULE_INFOS,
        MODULE_KEY_TO_MODULE_INFO_MAP,
        MODULE_LABEL_TO_MODULE_INFO_MAP,
    )
except Exception:
    traceback.print_exc()
    print("\nPlease update your library with 'python -m pip install -U --user .'")
    sys.exit(1)


if "win32" in sys.platform.lower():
    import ctypes

    myappid = "acconeer.exploration.tool"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

if __name__ == "__main__":
    if lib_version_up_to_date():
        utils.config_logging(level=logging.INFO)

        # Enable warnings to be printed to the log, e.g. DeprecationWarning
        warnings.simplefilter("module")

        app = QApplication(sys.argv)
        ex = GUI()

        signal.signal(signal.SIGINT, lambda *_: sigint_handler(ex))

        # Makes sure the signal is caught
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(200)

        sys.exit(app.exec_())
