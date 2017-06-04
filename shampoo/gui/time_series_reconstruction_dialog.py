from __future__ import absolute_import

from pyqtgraph import QtGui, QtCore, ImageView

from .recon_params_widget import ReconstructionParametersWidget
from .. import TimeSeries

class TimeSeriesReconstructionDialog(QtGui.QDialog):

    time_series_reconstructed = QtCore.pyqtSignal(str)
    time_series_loaded = QtCore.pyqtSignal(object)
    _reconstruction_update_signal = QtCore.pyqtSignal(int)

    def __init__(self, **kwargs):
        super(TimeSeriesReconstructionDialog, self).__init__(**kwargs)

        self.time_series = None
        self._propagation_distances = None
        self._fourier_mask = None

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

        progress_label = QtGui.QLabel('<h3>Reconstruction Progress</h3>')
        progress_label.setAlignment(QtCore.Qt.AlignCenter)

        self.reconstruction_progress = QtGui.QProgressBar(parent = self)
        self.reconstruction_progress.setRange(0, 100)
        self.reconstruction_progress.setValue(0)
        self.reconstruction_progress.setAlignment(QtCore.Qt.AlignCenter)
        self._reconstruction_update_signal.connect(self.reconstruction_progress.setValue)
        #self.reconstruction_progress.hide()

        self.recons_params_widget = ReconstructionParametersWidget(parent = self)
        self.recons_params_widget.propagation_distance_signal.connect(self.update_propagation_distance)
        self.recons_params_widget.fourier_mask_signal.connect(self.update_fourier_mask)

        load_time_series_btn = QtGui.QPushButton('Load time-series', self)
        load_time_series_btn.clicked.connect(self.load_time_series)

        accept_btn = QtGui.QPushButton('Reconstruct time-series', self)
        accept_btn.clicked.connect(self.accept)

        cancel_btn = QtGui.QPushButton('Cancel', self)
        cancel_btn.clicked.connect(self.reject)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(accept_btn)
        btns.addWidget(cancel_btn)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(load_time_series_btn)
        layout.addWidget(self.holograms_viewer)
        layout.addLayout(labeled_slider)
        layout.addWidget(self.recons_params_widget)
        layout.addWidget(progress_label)
        layout.addWidget(self.reconstruction_progress)
        layout.addLayout(btns)
        self.setLayout(layout)

    @QtCore.pyqtSlot()
    def load_time_series(self):
        path = QtGui.QFileDialog.getOpenFileName(self, caption = 'Select an HDF5 hologram time-series',
                                                 filter = '*.hdf5 *.h5')[0]
        
        if not path:
            return
        
        self.time_series = TimeSeries(name = path, mode = 'r+')

        self.holograms_slider.setRange(0, len(self.time_series.time_points) - 1)
        self.holograms_slider.setEnabled(True)
        self.update_holograms_viewer(index = 0)
    
    @QtCore.pyqtSlot(int)
    def update_holograms_viewer(self, index):
        """ Update the holograms viewer by time-point index """
        time_point = self.time_series.time_points[index]
        self.time_point_label.setNum(time_point)
        hologram = self.time_series.hologram(time_point)
        self.holograms_viewer.setImage(hologram.hologram)
    
    @QtCore.pyqtSlot(object)
    def update_propagation_distance(self, dist):
        self._propagation_distances = dist
    
    @QtCore.pyqtSlot(object)
    def update_fourier_mask(self, mask):
        self._fourier_mask = mask
    
    @QtCore.pyqtSlot()
    def accept(self):
        self.time_series.batch_reconstruct(propagation_distance = self._propagation_distances, 
                                           fourier_mask = self._fourier_mask, 
                                           callback = self._reconstruction_update_signal.emit)
        self.time_series.close()
        super().accept()
    
    @QtCore.pyqtSlot()
    def reject(self):
        if self.time_series:
            self.time_series.close()
        super().reject()
