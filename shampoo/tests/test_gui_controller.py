from __future__ import absolute_import

import os
import tempfile

import numpy as np
from pyqtgraph import QtCore

from .. import Hologram, ReconstructedWave, TimeSeries
from ..gui.controller import QReconstructor
from .test_hologram import _example_hologram


def test_qreconstructor_hologram_single_wavelength():
	reconstructed = list()

	app = QtCore.QCoreApplication([])

	quit_timer = QtCore.QTimer()
	quit_timer.timeout.connect(app.quit)

	# Test will hang forever if reconstructed_signal is directly connected
	# to app.quit
	# TODO: why??
	reconstructor = QReconstructor()
	reconstructor.reconstructed_signal.connect(reconstructed.append)
	reconstructor.reconstructed_signal.connect(lambda x: quit_timer.start(1))

	h = Hologram(_example_hologram())
	reconstructor.reconstruct(h, {'propagation_distance': 0.03})
	app.exec_() # Wait here until app.quit is called

	assert isinstance(reconstructed[0], ReconstructedWave)
	assert np.all(np.isfinite(reconstructed[0].reconstructed_wave))

def test_qreconstructor_hologram_three_wavelength():
	reconstructed = list()

	app = QtCore.QCoreApplication([])

	quit_timer = QtCore.QTimer()
	quit_timer.timeout.connect(app.quit)

	# Test will hang forever if reconstructed_signal is directly connected
	# to app.quit
	# TODO: why??
	reconstructor = QReconstructor()
	reconstructor.reconstructed_signal.connect(reconstructed.append)
	reconstructor.reconstructed_signal.connect(lambda x: quit_timer.start(1))

	h = Hologram(_example_hologram(), wavelength = [405e-9, 488e-9, 532e-9])
	reconstructor.reconstruct(h, {'propagation_distance': 0.03})
	app.exec_() # Wait here until app.exit is called

	assert isinstance(reconstructed[0], ReconstructedWave)
	assert np.all(np.isfinite(reconstructed[0].reconstructed_wave))
	assert reconstructed[0].reconstructed_wave.ndim == 3

def test_qreconstructor_on_time_series():
	name = os.path.join(tempfile.gettempdir(), 'test_time_series.hdf5')
	with TimeSeries(name = name, mode = 'w') as time_series:

		time_series.add_hologram(Hologram(_example_hologram()), 0)
		reconstructed = list()

		app = QtCore.QCoreApplication([])

		quit_timer = QtCore.QTimer()
		quit_timer.timeout.connect(app.quit)

		# Test will hang forever if reconstructed_signal is directly connected
		# to app.quit
		# TODO: why??
		reconstructor = QReconstructor()
		reconstructor.reconstructed_signal.connect(reconstructed.append)
		reconstructor.reconstructed_signal.connect(lambda x: quit_timer.start(1))

		reconstructor.reconstruct(time_series, {'propagation_distance': 0.03, 'time_point': 0})
		app.exec_() # Wait here until app.exit is called

		assert isinstance(reconstructed[0], ReconstructedWave)
		assert np.all(np.isfinite(reconstructed[0].reconstructed_wave))
