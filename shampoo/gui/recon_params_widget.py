from __future__ import absolute_import

import numpy as np
from pyqtgraph import QtGui, QtCore
from skimage.io import imread

from .fourier_mask_design_dialog import FourierMaskDesignDialog

from ..reconstruction import Hologram
from ..time_series import TimeSeries

DEFAULT_PROPAGATION_DISTANCE = 0.03658

class ReconstructionParametersWidget(QtGui.QWidget):

    propagation_distance_signal = QtCore.pyqtSignal(object)
    fourier_mask_signal = QtCore.pyqtSignal(object)
    chromatic_shift_signal = QtCore.pyqtSignal(tuple)

    _fourier_mask_path_signal = QtCore.pyqtSignal(str)

    def __init__(self, nwavelengths, hologram = None, *args, **kwargs):
        """
        Parameters
        ----------
        nwavelengths : int, {1, 3}
            Single or three-wavelength mode
        hologram : `~numpy.ndarray` or None, optional
            Image of a hologram.
        """
        if not nwavelengths in set([1, 3]):
            raise ValueError('Only single- and three-wavelengths reconstructions are available.')

        super(ReconstructionParametersWidget, self).__init__(*args, **kwargs)

        self.nwavelengths = nwavelengths
        self.hologram = hologram
        
        self.prop_dist_start_widget = QtGui.QDoubleSpinBox(parent = self)
        self.prop_dist_stop_widget = QtGui.QDoubleSpinBox(parent = self)
        self.prop_dist_step_widget = QtGui.QDoubleSpinBox(parent = self)

        self.prop_dist_start_label = QtGui.QLabel('Start')
        self.prop_dist_start_label.setAlignment(QtCore.Qt.AlignCenter)
        self.prop_dist_stop_label = QtGui.QLabel('Stop')
        self.prop_dist_stop_label.setAlignment(QtCore.Qt.AlignCenter)
        self.prop_dist_step_label = QtGui.QLabel('Step')
        self.prop_dist_step_label.setAlignment(QtCore.Qt.AlignCenter)
        
        for widget in (self.prop_dist_start_widget, 
                       self.prop_dist_stop_widget, 
                       self.prop_dist_step_widget):
            widget.setSuffix(' m')
            widget.setRange(0.0, 1.0)
            widget.setDecimals(5)
            widget.setSingleStep(1e-5)
            widget.editingFinished.connect(self.update_propagation_distance)
        
        self.prop_dist_start_widget.setValue(DEFAULT_PROPAGATION_DISTANCE)
        self.set_multi_dist_mode(mode = 0)  # Initialize to single-propagation distance
        
        # Button group for single or multiple propagation distances
        single_prop_dist_btn = QtGui.QPushButton('Single propagation distance', parent = self)
        single_prop_dist_btn.setCheckable(True)
        single_prop_dist_btn.setChecked(True)
        multi_prop_dist_btn = QtGui.QPushButton('Multiple propagation distances', parent = self)
        multi_prop_dist_btn.setCheckable(True)
        multi_prop_dist_btn.setChecked(False)

        propagation_distance_mode = QtGui.QButtonGroup(parent = self)
        propagation_distance_mode.addButton(single_prop_dist_btn, 0)
        propagation_distance_mode.addButton(multi_prop_dist_btn, 1)
        propagation_distance_mode.setExclusive(True)
        propagation_distance_mode.buttonClicked[int].connect(self.set_multi_dist_mode)

        prop_dist_layout = QtGui.QGridLayout()
        prop_dist_layout.addWidget(self.prop_dist_start_label, 0, 0)
        prop_dist_layout.addWidget(self.prop_dist_step_label, 0, 1)
        prop_dist_layout.addWidget(self.prop_dist_stop_label, 0, 2)
        prop_dist_layout.addWidget(self.prop_dist_start_widget, 1, 0)
        prop_dist_layout.addWidget(self.prop_dist_step_widget, 1, 1)
        prop_dist_layout.addWidget(self.prop_dist_stop_widget, 1, 2)

        mode_btns = QtGui.QHBoxLayout()
        mode_btns.addWidget(single_prop_dist_btn)
        mode_btns.addWidget(multi_prop_dist_btn)

        propagation_distances_label = QtGui.QLabel('<h3>Numerical Propagation Distance(s)</h3>')
        propagation_distances_label.setAlignment(QtCore.Qt.AlignCenter)

        self.chromshift1_widget = QtGui.QDoubleSpinBox(parent = self)
        self.chromshift2_widget = QtGui.QDoubleSpinBox(parent = self)
        self.chromshift3_widget = QtGui.QDoubleSpinBox(parent = self)

        for widget in (self.chromshift1_widget, self.chromshift2_widget, self.chromshift3_widget):
            widget.setSuffix(' m')
            widget.setRange(-1.0, 1.0)
            widget.setSingleStep(1e-6)
            widget.setDecimals(6)
            widget.setValue(0.0)
            widget.valueChanged.connect(self.update_chromatic_shift)
        
        chromshift_layout = QtGui.QHBoxLayout()
        chromshift_layout.addWidget(self.chromshift1_widget)
        chromshift_layout.addWidget(self.chromshift2_widget)
        chromshift_layout.addWidget(self.chromshift3_widget)

        if self.nwavelengths == 1:
            self.chromshift2_widget.hide()
            self.chromshift3_widget.hide()

        chromatic_shifts_label = QtGui.QLabel('<h3>Chromatic Shift(s)</h3>')
        chromatic_shifts_label.setAlignment(QtCore.Qt.AlignCenter)

        self.load_fourier_mask_btn = QtGui.QPushButton('Load Fourier mask', self)
        self.load_fourier_mask_btn.clicked.connect(self.load_fourier_mask)

        self.design_fourier_mask_btn = QtGui.QPushButton('Design Fourier mask', self)
        self.design_fourier_mask_btn.clicked.connect(self.design_fourier_mask)

        self.clear_fourier_mask_btn = QtGui.QPushButton('Clear Fourier mask', self)
        self.clear_fourier_mask_btn.clicked.connect(self.clear_fourier_mask)

        self.fourier_mask_path_label = QtGui.QLabel('Current Fourier mask: None', self)
        self.fourier_mask_path_label.setAlignment(QtCore.Qt.AlignCenter)
        self._fourier_mask_path_signal.connect(self.fourier_mask_path_label.setText)

        fourier_mask_btns = QtGui.QHBoxLayout()
        fourier_mask_btns.addWidget(self.load_fourier_mask_btn)
        fourier_mask_btns.addWidget(self.design_fourier_mask_btn)
        fourier_mask_btns.addWidget(self.clear_fourier_mask_btn)

        fourier_mask_label = QtGui.QLabel('<h3>Fourier Mask Controls</h3>')
        fourier_mask_label.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(propagation_distances_label)
        layout.addLayout(prop_dist_layout)
        layout.addLayout(mode_btns)
        layout.addWidget(chromatic_shifts_label)
        layout.addLayout(chromshift_layout)
        layout.addWidget(fourier_mask_label)
        layout.addWidget(self.fourier_mask_path_label)
        layout.addLayout(fourier_mask_btns)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(int)
    def set_multi_dist_mode(self, mode):
        """ Change multi-propagation distance mode. If id = 0, single-propagation
        distance mode. if id = 1: multi-propagation distance mode. """
        if mode == 0:
            self.prop_dist_start_label.setText('Distance')
            self.prop_dist_step_label.hide()
            self.prop_dist_stop_label.hide()
            self.prop_dist_stop_widget.hide()
            self.prop_dist_step_widget.hide()
            self.prop_dist_step_widget.setValue(0.0)
        elif mode == 1:
            self.prop_dist_start_label.setText('Start')
            self.prop_dist_step_label.show()
            self.prop_dist_stop_label.show()
            self.prop_dist_stop_widget.show()
            self.prop_dist_step_widget.show()
            self.prop_dist_step_widget.setValue(0.0)

    @QtCore.pyqtSlot()
    def load_fourier_mask(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Load Fourier-domain mask', filter = '*.tif')[0]
        if path:
            self.fourier_mask_signal.emit(imread(path))
            self._fourier_mask_path_signal.emit('Current Fourier mask: {}'.format(path)) 
    
    @QtCore.pyqtSlot()
    def design_fourier_mask(self):
        ft = np.fft.fftshift(np.fft.fft2(self.hologram))
        designer = FourierMaskDesignDialog(parent = self, fourier = np.log(np.abs(ft)**2))
        designer.fourier_mask.connect(self.fourier_mask_signal)
        designer.fourier_mask.connect(lambda im: self._fourier_mask_path_signal.emit('User-designed mask'))
        success = designer.exec_()
    
    @QtCore.pyqtSlot()
    def clear_fourier_mask(self):
        self.fourier_mask_signal.emit(None)
        self._fourier_mask_path_signal.emit('Current Fourier mask: None')
    
    @QtCore.pyqtSlot()
    def update_chromatic_shift(self):
        if self.nwavelengths == 1:
            self.chromatic_shift_signal.emit(tuple([self.chromshift1_widget.value()]))
        else:
            self.chromatic_shift_signal.emit(tuple([self.chromshift1_widget.value(),
                                                    self.chromshift2_widget.value(),
                                                    self.chromshift3_widget.value()]))
    
    @QtCore.pyqtSlot()
    def update_propagation_distance(self):
        """ Emits the propagation_distance signal with the sorted 
        propagation distance data parsed from the widget. """

        start, stop, step = (self.prop_dist_start_widget.value(), 
                             self.prop_dist_stop_widget.value(),
                             self.prop_dist_step_widget.value())
        
        propagation_distance = np.array([start]) if step == 0 else np.arange(start = start, 
                                                                           stop = stop + step, 
                                                                           step = step)
        self.propagation_distance_signal.emit(propagation_distance)