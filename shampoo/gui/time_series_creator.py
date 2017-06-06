from __future__ import absolute_import

from pyqtgraph import QtGui, QtCore

from ..reconstruction import Hologram
from ..time_series import TimeSeries

class TimeSeriesCreator(QtGui.QDialog):

    time_series_path_signal = QtCore.pyqtSignal(str)
    _assembly_update_signal = QtCore.pyqtSignal(int)

    time_series_assembly = QtCore.pyqtSignal(dict)

    def __init__(self, **kwargs):
        self.holograms = list()

        super(TimeSeriesCreator, self).__init__(**kwargs)
        self.setModal(True)
        self.setWindowTitle('Create hologram time series')

        # Create reorderable list
        # Example from http://www.walletfox.com/course/qtreorderablelist.php
        self.hologram_table = QtGui.QListWidget(parent = self)
        self.hologram_table.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        
        self.assembly_progress_bar = QtGui.QProgressBar(parent = self)
        self.assembly_progress_bar.setRange(0, 100)
        self.assembly_progress_bar.setValue(0)
        self._assembly_update_signal.connect(self.assembly_progress_bar.setValue)

        progress_label = QtGui.QLabel('<h3>Assembly Progress</h3>')
        progress_label.setAlignment(QtCore.Qt.AlignCenter)

        # Wavelength widgets as spinboxes
        # wavelength 2 and 3 are hidden with a default of None
        self.wavelength1_widget = QtGui.QSpinBox(parent = self)
        self.wavelength2_widget = QtGui.QSpinBox(parent = self)
        self.wavelength3_widget = QtGui.QSpinBox(parent = self)

        self.wavelength2_widget.hide()
        self.wavelength3_widget.hide()

        for widget in (self.wavelength1_widget, self.wavelength2_widget, self.wavelength3_widget):
            widget.setSuffix(' nm')
            widget.setMinimum(0)    # value of 0  -> not to be counted
            widget.setMaximum(999)
        
        self.wavelength1_widget.setValue(405)
        self.wavelength1_widget.setMinimum(1)   # At least one wavelength must be given

        # Create an exclusive button group in which only one-wavelength or three-wavelengths
        # can be active at one time
        self.one_wavelength_mode_btn = QtGui.QPushButton('Single-wavelength time-series', self)
        self.one_wavelength_mode_btn.setCheckable(True)
        self.one_wavelength_mode_btn.setChecked(True)
        self.three_wavelength_mode_btn = QtGui.QPushButton('Three-wavelength time-series', self)
        self.three_wavelength_mode_btn.setCheckable(True)
        self.three_wavelength_mode_btn.setChecked(False)

        self.wavelength_btns = QtGui.QButtonGroup(parent = self)
        self.wavelength_btns.addButton(self.one_wavelength_mode_btn, id = 1)
        self.wavelength_btns.addButton(self.three_wavelength_mode_btn, id = 3)
        self.wavelength_btns.setExclusive(True)
        self.wavelength_btns.buttonClicked[int].connect(self.set_wavelength_mode)

        file_search_btn = QtGui.QPushButton('Add hologram', self)
        file_search_btn.clicked.connect(self.add_hologram_file)

        clear_btn = QtGui.QPushButton('Clear holograms', self)
        clear_btn.clicked.connect(self.clear)

        accept_btn = QtGui.QPushButton('Create', self)
        accept_btn.clicked.connect(self.accept)

        reject_btn = QtGui.QPushButton('Cancel', self)
        reject_btn.clicked.connect(self.reject)
        reject_btn.setDefault(True)

        hologram_btns = QtGui.QHBoxLayout()
        hologram_btns.addWidget(file_search_btn)
        hologram_btns.addWidget(clear_btn)

        wavelength_mode_layout = QtGui.QHBoxLayout()
        wavelength_mode_layout.addWidget(self.one_wavelength_mode_btn)
        wavelength_mode_layout.addWidget(self.three_wavelength_mode_btn)

        wavelength_layout = QtGui.QHBoxLayout()
        wlabel = QtGui.QLabel('Wavelength(s):')
        wlabel.setAlignment(QtCore.Qt.AlignCenter)
        wavelength_layout.addWidget(wlabel)
        wavelength_layout.addWidget(self.wavelength1_widget)
        wavelength_layout.addWidget(self.wavelength2_widget)
        wavelength_layout.addWidget(self.wavelength3_widget)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(accept_btn)
        btns.addWidget(reject_btn)

        explanation = QtGui.QLabel('Add holograms and order them by drag-and-drop')
        explanation.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(explanation)
        layout.addLayout(hologram_btns)
        layout.addWidget(self.hologram_table)
        layout.addLayout(wavelength_mode_layout)
        layout.addLayout(wavelength_layout)
        layout.addWidget(progress_label)
        layout.addWidget(self.assembly_progress_bar)
        layout.addLayout(btns)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot()
    def clear(self):
        self.holograms.clear()
        self.hologram_table.clear()
    
    @QtCore.pyqtSlot(int)
    def set_wavelength_mode(self, n_wavelengths):
        """ Change the context to build a time-series with n_wavelengths. """
        if n_wavelengths == 1:
            self.wavelength2_widget.hide()
            self.wavelength2_widget.setValue(0)
            self.wavelength3_widget.hide()
            self.wavelength3_widget.setValue(0)
        
        elif n_wavelengths == 3:
            self.wavelength2_widget.show()
            self.wavelength2_widget.setValue(488)
            self.wavelength3_widget.show()
            self.wavelength3_widget.setValue(532)
    
    @QtCore.pyqtSlot()
    def add_hologram_file(self):
        """ 
        Add hologram to the holograms dictionary by filepath. 

        Parameters
        ----------
        path : str or iterable of strings
            Path or paths to a hologram.
        """
        paths = QtGui.QFileDialog.getOpenFileNames(self, caption = 'Select one of more holograms', 
                                                   filter = "Images (*.tif)")[0]
        self.hologram_table.addItems(paths)
        self.holograms.extend(paths)
        
    @QtCore.pyqtSlot()
    def accept(self):
        assembly_params = dict()
        filename = QtGui.QFileDialog.getSaveFileName(self, caption = 'Save time series', 
                                                   filter = "HDF5 (*.hdf5 *.h5)")[0]
        if not filename:
            return

        # Determine the number of wavelengths
        # wavelengths of value 0 are not counted.
        wavelengths = list()
        for widget in (self.wavelength1_widget, self.wavelength2_widget, self.wavelength3_widget):
            v = widget.value()
            if v != 0:
                wavelengths.append(v*1e-9)  # widgets show nm, we want meters

        assembly_params['filename'] = filename
        assembly_params['callback'] = self._assembly_update_signal.emit
        assembly_params['wavelengths'] = wavelengths
        assembly_params['hologram_paths'] = list(self.holograms)

        # NOTE: even though the arguments for super() are not required in python 3, they are required
        #       when emitting this dictionary away
        assembly_params['final_callback'] = lambda : super(TimeSeriesCreator, self).accept()

        self.time_series_assembly.emit(assembly_params)