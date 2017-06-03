from __future__ import absolute_import

import numpy as np
from pyqtgraph import QtGui, QtCore

from ..reconstruction import Hologram
from ..time_series import TimeSeries

DEFAULT_PROPAGATION_DISTANCE = 0.03658

class ReconstructionParametersWidget(QtGui.QWidget):

    propagation_distance_signal = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super(ReconstructionParametersWidget, self).__init__(*args, **kwargs)
        
        self.prop_dist_start_widget = QtGui.QDoubleSpinBox(parent = self)
        self.prop_dist_start_widget.setValue(DEFAULT_PROPAGATION_DISTANCE)
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
            widget.setSingleStep(10^-(widget.decimals()))
            widget.setDecimals(5)
            widget.editingFinished.connect(self.update_propagation_distance)
        
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

        title_label = QtGui.QLabel('<h3>Numerical Propagation Distance(s)</h3>')
        title_label.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(title_label)
        layout.addLayout(prop_dist_layout)
        layout.addLayout(mode_btns)
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