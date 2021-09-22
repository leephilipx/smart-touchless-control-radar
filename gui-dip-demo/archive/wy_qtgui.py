import sys
from pyqtgraph.Qt import QtGui

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 800, 500)
        self.setWindowTitle("PyQT tuts!")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))
        self.add()

    def add(self):
        
        btnStart = QtGui.QPushButton("Start", self)
        btnStart.clicked.connect(self.btn_start)
        btnStart.move(100,0)
        btnNext = QtGui.QPushButton("Next", self)
        btnNext.clicked.connect(self.btn_start)
        btnNext.move(200,0)
        btnDiscard = QtGui.QPushButton("Discard", self)
        btnDiscard.clicked.connect(self.btn_start)
        btnDiscard.move(300,0)
        btnStop = QtGui.QPushButton("Stop", self)
        btnStop.clicked.connect(self.btn_start)
        btnStop.move(400,0)
        self.txtFilename = QtGui.QLineEdit(".npy", self)
        self.txtFilename.setCursorPosition(0)
        self.txtFilename.move(500, 0)
        self.txtFilename.resize(150, 25)

        self.count = QtGui.QLabel("Count=0", self)
        self.count.move(700, 0)
        self.count.resize(150, 25)
        self.txtFilename.setFocus()
        self.show()

    def btn_start(self):
        pass
app = QtGui.QApplication(sys.argv)
GUI = Window()
sys.exit(app.exec_())