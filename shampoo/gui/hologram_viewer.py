from __future__ import absolute_import

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore

from ..reconstruction import Hologram, fft2, fftshift
from ..time_series import TimeSeries

class HologramViewer(QtGui.QWidget):
    """
    QWidget displaying raw holograms, either from TIFFs or TimeSeries, 
    as well as related information such as Fourier decomposition.

    Slots
    -----
    display
        Display a Hologram object
    
    toggle_time_series_controls
        Show/hide time-series controls, such as time-point selectors
    """

    def __init__(self, *args, **kwargs):
        super(HologramViewer, self).__init__(*args, **kwargs)

        self.raw_data_viewer = pg.ImageView(parent = self, name = 'Raw data')
        self.fourier_plane_viewer = pg.ImageView(parent = self, name = 'Fourier plane')

        tabs = QtGui.QTabWidget(parent = self)
        tabs.addTab(self.raw_data_viewer, 'Raw hologram')
        tabs.addTab(self.fourier_plane_viewer, 'Fourier plane')

        layout = QtGui.QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(object)
    def display(self, data):
        """
        Display raw hologram and associated Fourier plane information.

        Parameters
        ----------
        data : Hologram
        """
        self.raw_data_viewer.setImage(np.squeeze(data.hologram))

        ft = fftshift(fft2(np.squeeze(data.hologram), axes = (0, 1)), axes = (0, 1))
        self.fourier_plane_viewer.setImage(ft.real)
