from __future__ import absolute_import

from pyqtgraph import QtGui, QtCore, ImageView

from .. import TimeSeries

class TimeSeriesReconstructionDialog(QtGui.QDialog):

    time_series_reconstructed = QtCore.pyqtSignal(str)
    time_series_loaded = QtCore.pyqtSignal(object)
    _reconstruction_update_signal = QtCore.pyqtSignal(int)

    def __init__(self, **kwargs):
        super(TimeSeriesReconstructionDialog, self).__init__(**kwargs)

        self.time_series = None

        self.setModal(True)
        self.setWindowTitle('Reconstruct time-series')

        self.holograms_viewer = ImageView(parent = self, name = 'Raw holograms')

        self.holograms_slider = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.holograms_slider.setDisabled(True)
        self.holograms_slider.valueChanged.connect(self.update_holograms_viewer)

        self.time_point_label = QtGui.QLabel('')
        self.time_point_label.setAlignment(QtCore.Qt.AlignCenter)

        labeled_slider = QtGui.QHBoxLayout()
        labeled_slider.addWidget(self.time_point_label)
        labeled_slider.addWidget(self.holograms_slider)

        self.reconstruction_progress = QtGui.QProgressBar(parent = self)
        self.reconstruction_progress.setRange(0, 100)
        self._reconstruction_update_signal.connect(self.reconstruction_progress.setValue)
        self.reconstruction_progress.hide()

        load_time_series_btn = QtGui.QPushButton('Load time-series', self)
        load_time_series_btn.clicked.connect(self.load_time_series)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.holograms_viewer)
        layout.addLayout(labeled_slider)
        layout.addWidget(load_time_series_btn)
        self.setLayout(layout)

    @QtCore.pyqtSlot()
    def load_time_series(self):
        path = QtGui.QFileDialog.getOpenFileName(self, caption = 'Select an HDF5 hologram time-series',
                                                 filter = '*.hdf5 *.h5')[0]
        
        if path == '':
            return
        
        self.time_series = TimeSeries(name = path, mode = 'r+')

        # Fill in the details
        self.holograms_slider.setRange(0, len(self.time_series.time_points))
        self.holograms_slider.setEnabled(True)
        self.update_holograms_viewer(index = 0)
    
    @QtCore.pyqtSlot(int)
    def update_holograms_viewer(self, index):
        """ Update the holograms viewer by time-point index """
        time_point = self.time_series.time_points[index]
        self.time_point_label.setNum(time_point)
        hologram = self.time_series.hologram(time_point)
        self.holograms_viewer.setImage(hologram.hologram)
    
    @QtCore.pyqtSlot()
    def accept(self):
        raise NotImplementedError
        self.time_series.batch_reconstruct(propagation_distance = range(0,10), 
                                           fourier_mask = fourier_mask, 
                                           callback = self._reconstruction_update_signal.emit)
