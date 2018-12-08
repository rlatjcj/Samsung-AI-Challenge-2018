from PyQt5 import QtGui, QtCore

import sys
import pyqtgraph
import numpy as np
import windows
import audio
import time

import client
import model

HOST = '169.254.105.151'
PORT = 4006

class ExampleApp(QtGui.QMainWindow, windows.Ui_MainWindow):
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'w')
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.pcm.plotItem.showGrid(True, True, 0.7)
        self.nc_pcm.plotItem.showGrid(True, True, 0.7)

        self.maxPCM = 0
        self.maxnc_PCM = 0

        self.rate = 44100

        self.ear = audio.AUDIO(rate=self.rate, updatesPerSecond=100)
        self.ear.stream_start()

        self.model = model


    def update(self):
        if not self.ear.data is None and not self.ear.bpf is None:

            pcmMax = np.max(np.abs(self.ear.data))
            if pcmMax > self.maxPCM:
                self.maxPCM = pcmMax
##                self.pcm.plotItem.setRange(yRange=[-pcmMax,pcmMax])
                self.pcm.plotItem.setRange(yRange=[-1000,1000])

            nc_pcmMax = np.max(np.abs(self.ear.bpf))
            if nc_pcmMax > self.maxnc_PCM:
                self.maxnc_PCM = nc_pcmMax
                self.nc_pcm.plotItem.setRange(yRange=[-1000,1000])


            pen = pyqtgraph.mkPen(color='r')
            self.pcm.plot(self.ear.datax, self.ear.data, pen=pen, clear=True)
            pen = pyqtgraph.mkPen(color='k')
            self.nc_pcm.plot(self.ear.datax, -self.ear.bpf, pen=pen, clear=True)

        QtCore.QTimer.singleShot(1, self.update) # QUICKLY repeat

    def send_click(self):
        print('override send function click')

        #self.client.run()
        #getting 10 sec sound data
        x_data = self.ear.pop_all_q()

        #send it to server
        y_pred = self.model.run_model(x_data)

        #remove others category
        self.data = self.filtering_others(y_pred)

        for row, d in enumerate(self.data):
            for column in range(2):
                item = QtGui.QTableWidgetItem(d)
                chbox = QtGui.QTableWidgetItem()
                chbox.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                chbox.setCheckState(QtCore.Qt.Unchecked)
                self.nc_list.setItem(row, 0, chbox)
                self.nc_list.setItem(row, 1, item)
                self.nc_list.setColumnWidth(0, 40)
                self.nc_list.setColumnWidth(1, 400)
        self.nc_list.itemClicked.connect(self.handleItemClicked)

    def filtering_others(self, arr):
        if 'others' in arr:
            print('there is others category')
            arr.remove('others')
            return arr
        else :
            print('there is no others category')
            return arr

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    form.update()
    #form.client.connect_server()
    #print('connec')
    app.exec_()
    print("DONE")
