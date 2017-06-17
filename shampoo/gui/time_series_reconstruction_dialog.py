from __future__ import absolute_import

from pyqtgraph import QtGui, QtCore

from .recon_params_widget import ReconstructionParametersWidget
from .. import TimeSeries

class TimeSeriesReconstructionDialog(QtGui.QDialog):

    reconstruction_parameters = QtCore.pyqtSignal(dict)
    _reconstruction_update_signal = QtCore.pyqtSignal(int)

    def __init__(self, nwavelengths, **kwargs):
        """
        Parameters
        ----------
        nwavelengths : int, {1, 3}
            Single or three-wavelength mode
        """
        super(TimeSeriesReconstructionDialog, self).__init__(**kwargs)

        self._propagation_distances = None
        self._fourier_mask = None
        self._chromatic_shifts = None

        self.setModal(True)
        self.setWindowTitle('Reconstruct time-series')

        progress_label = QtGui.QLabel('<h3>Reconstruction Progress</h3>')
        progress_label.setAlignment(QtCore.Qt.AlignCenter)

        self.reconstruction_progress = QtGui.QProgressBar(parent = self)
        self.reconstruction_progress.setRange(0, 100)
        self.reconstruction_progress.setValue(0)
        self.reconstruction_progress.setAlignment(QtCore.Qt.AlignCenter)
        self._reconstruction_update_signal.connect(self.reconstruction_progress.setValue)

        self.recons_params_widget = ReconstructionParametersWidget(parent = self, nwavelengths = nwavelengths)
        self.recons_params_widget.propagation_distance_signal.connect(self.update_propagation_distance)
        self.recons_params_widget.fourier_mask_signal.connect(self.update_fourier_mask)
        self.recons_params_widget.chromatic_shift_signal.connect(self.update_chromatic_shift)

        accept_btn = QtGui.QPushButton('Reconstruct time-series', self)
        accept_btn.clicked.connect(self.accept)

        cancel_btn = QtGui.QPushButton('Cancel', self)
        cancel_btn.clicked.connect(self.reject)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(accept_btn)
        btns.addWidget(cancel_btn)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.recons_params_widget)
        layout.addWidget(progress_label)
        layout.addWidget(self.reconstruction_progress)
        layout.addLayout(btns)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(object)
    def update_propagation_distance(self, dist):
        self._propagation_distances = dist
    
    @QtCore.pyqtSlot(object)
    def update_fourier_mask(self, mask):
        self._fourier_mask = mask
    
    @QtCore.pyqtSlot(tuple)
    def update_chromatic_shift(self, shifts):
        self._chromatic_shifts = shifts
    
    @QtCore.pyqtSlot()
    def accept(self):
        recon_params = {'propagation_distance': self._propagation_distances,
                        'fourier_mask': self._fourier_mask,
                        'chromatic_shift': self._chromatic_shifts,
                        'callback': self._reconstruction_update_signal.emit,
                        'final_callback': lambda: super(TimeSeriesReconstructionDialog, self).accept()}
        self.reconstruction_parameters.emit(recon_params)
    
    @QtCore.pyqtSlot()
    def reject(self):
        super().reject()
