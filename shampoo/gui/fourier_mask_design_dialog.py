from pyqtgraph import QtGui, QtCore, ImageView, CircleROI, mkPen
import numpy as np

class FourierMaskDesignDialog(QtGui.QDialog):

    fourier_mask = QtCore.pyqtSignal(object)

    def __init__(self, fourier = None, **kwargs):
        """
        Parameters
        ----------
        fourier : `~numpy.ndarray` or None, optional
            If an array, the log-scale intensity of the `fourier` array will be displayed as 
            an image and the mask can be designed on top of it.
        """
        super(FourierMaskDesignDialog, self).__init__(**kwargs)
        self.setModal(True)
        self.setWindowTitle('Design a Fourier mask')

        self.viewer = ImageView(parent = self)
        #self.viewer.setColorMap()
        if fourier is not None:
            self.viewer.setImage(np.log(np.abs(fourier)**2))
        
        self.rois = list()  # roi = region of interest
        
        add_roi_btn = QtGui.QPushButton('Add shape', self)
        add_roi_btn.clicked.connect(self.add_roi)

        clear_roi_btn = QtGui.QPushButton('Clear all shapes', self)
        clear_roi_btn.clicked.connect(self.clear_rois)

        accept_btn = QtGui.QPushButton('Accept', self)
        accept_btn.clicked.connect(self.accept)

        cancel_btn = QtGui.QPushButton('Cancel', self)
        cancel_btn.clicked.connect(self.reject)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(add_roi_btn)
        btns.addWidget(clear_roi_btn)
        btns.addWidget(accept_btn)
        btns.addWidget(cancel_btn)

        explanation = QtGui.QLabel('Add explanation here')

        layout = QtGui.QVBoxLayout()
        layout.addWidget(explanation)
        layout.addWidget(self.viewer)
        layout.addLayout(btns)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot()
    def add_roi(self):
        """ Add an ROI to the viewer """
        new_roi = CircleROI(pos = [1024, 1024], size = [200, 200], pen = mkPen('r', width = 4))
        self.viewer.addItem(new_roi)
        self.rois.append(new_roi)
    
    @QtCore.pyqtSlot()
    def clear_rois(self):
        for roi in self.rois:
            self.viewer.removeItem(roi)
    
    @QtCore.pyqtSlot()
    def accept(self):
        # TODO: get shape from image in self.viewer
        mask = np.zeros( (2048, 2048), dtype = np.bool )
        xx, yy = np.meshgrid(np.linspace(-mask.shape[0]/2, mask.shape[0]/2), 
                             np.linspace(-mask.shape[1]/2, mask.shape[1]/2))
        rr = np.sqrt(xx**2 + yy**2)

        for roi in self.rois:
            radius = roi.size().x()/2
            corner_x, corner_y = roi.pos().x(), self.center_finder.pos().y()
            xc, yc = (round(corner_y + radius), round(corner_x + radius)) #Flip output since image viewer plots transpose...
            
            mask[ (rr - np.sqrt(xc**2 + yc**2)) < radius ] = True

        self.fourier_mask.emit(mask)
        super().accept()

    @QtCore.pyqtSlot()
    def reject(self):
        super().reject()
