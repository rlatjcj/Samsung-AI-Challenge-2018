from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from pyqtgraph import PlotWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1700, 1080)

        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))

        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))

        self.frame1 = QtGui.QFrame(self.centralwidget)
        self.frame1.setFrameShape(QtGui.QFrame.NoFrame)
        self.frame1.setFrameShadow(QtGui.QFrame.Plain)
        self.frame1.setObjectName(_fromUtf8("frame1"))
        self.verticalLayout1 = QtGui.QVBoxLayout(self.frame1)
        self.verticalLayout1.setObjectName(_fromUtf8("verticalLayout1"))

        # FFT
        self.label_6 = QtGui.QLabel(self.frame1)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.verticalLayout1.addWidget(self.label_6)
        self.fft = PlotWidget(self.frame1)
        self.fft.setObjectName(_fromUtf8("fft"))
        self.verticalLayout1.addWidget(self.fft)

        # PCM
        self.label_1 = QtGui.QLabel(self.frame1)
        self.label_1.setObjectName(_fromUtf8("label_1"))
        self.verticalLayout1.addWidget(self.label_1)
        self.pcm = PlotWidget(self.frame1)
        self.pcm.setObjectName(_fromUtf8("pcm"))
        self.verticalLayout1.addWidget(self.pcm)

        # PCM for NOISE CANCELLING
        self.label_2 = QtGui.QLabel(self.frame1)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout1.addWidget(self.label_2)
        self.nc_pcm = PlotWidget(self.frame1)
        self.nc_pcm.setObjectName(_fromUtf8("nc_pcm"))
        self.verticalLayout1.addWidget(self.nc_pcm)

        self.horizontalLayout.addWidget(self.frame1)

        self.frame2 = QtGui.QFrame(self.centralwidget)
        self.frame2.setFrameShape(QtGui.QFrame.NoFrame)
        self.frame2.setFrameShadow(QtGui.QFrame.Plain)
        self.frame2.setObjectName(_fromUtf8("frame2"))
        self.verticalLayout2 = QtGui.QVBoxLayout(self.frame2)
        self.verticalLayout2.setObjectName(_fromUtf8("verticalLayout2"))

        # SEND 10s to SERVER
        self.label_3 = QtGui.QLabel(self.frame2)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout2.addWidget(self.label_3)
        self.send_button = QPushButton("SEND", self.frame2)
        self.send_button.clicked.connect(self.send_click)
        self.verticalLayout2.addWidget(self.send_button)

        # NOISE LIST
        self.label_4 = QtGui.QLabel(self.frame2)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.verticalLayout2.addWidget(self.label_4)
        self.nc_list = QTableWidget(7, 2, self.frame2) # rows, columns, self
        self.nc_list.setHorizontalHeaderLabels(["Y/N", "NOISE"])
        self.nc_list.setColumnWidth(0, 40)
        self.nc_list.setColumnWidth(1, 400)
        self.nc_list.setObjectName(_fromUtf8("nc_list"))
        self.verticalLayout2.addWidget(self.nc_list)
        self.nc_button = QPushButton("CANCELLING", self.frame2)
        self.nc_button.clicked.connect(self.nc_click)
        self.verticalLayout2.addWidget(self.nc_button)

        # PROCESSING WINDOW
        self.label_5 = QtGui.QLabel(self.frame2)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout2.addWidget(self.label_5)
        # self.process_window = QListWidget()

        self.fft_figure = plt.figure()
        self.canvas_fft = FigureCanvas(self.fft_figure)
        self.verticalLayout2.addWidget(self.canvas_fft)
        self.bp_figure = plt.figure()
        self.canvas = FigureCanvas(self.bp_figure)
        self.verticalLayout2.addWidget(self.canvas)

        self.horizontalLayout.addWidget(self.frame2)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label_1.setText(_translate("MainWindow", "PCM (R: PCM / B: RESULT)", None))
        self.label_2.setText(_translate("MainWindow", "PCM for NOISE CANCELLING", None))
        self.label_3.setText(_translate("MainWindow", "SEND 10s to SERVER", None))
        self.label_4.setText(_translate("MainWindow", "NOISE LIST", None))
        self.label_5.setText(_translate("MainWindow", "PROCESSING WINDOW", None))
        self.label_6.setText(_translate("MainWindow", "FFT of PCM (R: RAW / B: RESULT)", None))

    def send_click(self):
        pass

    def handleItemClicked(self, item):
        try:
            if item.checkState() == QtCore.Qt.Checked:
                self._noiselist.append(self.data[item.row()])
                print(self._noiselist)
                print('{} Checked'.format(self.data[item.row()]))
            elif item.checkState() == QtCore.Qt.Unchecked:
                self._noiselist.remove(self.data[item.row()])
                print(self._noiselist)
                print('{} Unchecked'.format(self.data[item.row()]))
        except:
            print('click in checkbox!')
            pass

    def nc_click(self):
        pass
