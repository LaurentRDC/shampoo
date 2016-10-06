"""
Convenience functions and objects for SHAMPOO's GUI component.

Author: Laurent P. Rene de Cotret

Objects
-------
ComputationThread

ProgressWidget
"""
from math import sin, cos, pi
from PyQt4 import QtCore, QtGui

class ComputationThread(QtCore.QThread):
    """
    Simplified thread interface to be able to do operations
    without freezing the GUI.
    
    Signals
    -------
    done_signal
        Emitted when the function evaluation is over.
    in_progress_signal
        Emitted when the function evaluation starts.
    results_signal
        Emitted when the function evaluation is over. Carries the results
        of the computation.
        
    Attributes
    ----------
    function : callable
        Function to be called
    args : tuple
        Positional arguments of function
    kwargs : dict
        Keyword arguments of function
    results : object
        Results of the computation
    
    Examples
    --------
    >>> function = lambda x : x ** 10
    >>> result_function = lambda x: print(x)
    >>> worker = ComputationThread(function, 2)
    >>> worker.result_signal.connect(result_function)
    >>> worker.run()                                           # Computation starts only when this method is called
    """
    done_signal = QtCore.pyqtSignal(bool, name = 'done_signal')
    in_progress_signal = QtCore.pyqtSignal(bool, name = 'in_progress_signal')
    results_signal = QtCore.pyqtSignal(object, name = 'results_signal')
    
    def __init__(self, function, *args, **kwargs):
        """
        Parameters
        ----------
        function : callable
            Function to be called.
        args : tuple
            Positional arguments of function
        kwargs : dict
            Keyword arguments of function
        """
        super(ComputationThread, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def __del__(self):
        self.wait()
    
    def run(self):
        self.in_progress_signal.emit(True)
        self.result = self.function(*self.args, **self.kwargs)  # Computation is here      
        self.done_signal.emit(True)        
        self.results_signal.emit(self.result)


class InProgressWidget(QtGui.QWidget):
    """ 
    Spinning wheel with transparent background to overlay over other widgets during computations. 

    Notes
    -----
    When attaching this widget to another widget, it is important to oerride the resizeEvent of the
    parent to resize the progress widget as well.
    """
    # Number of dots to display as part of the 'spinning wheel'
    _num_points = 12

    def __init__(self, parent):
        """
        Parameters
        ----------
        parent : QWidget instance
        """
        super(InProgressWidget, self).__init__(parent)        
        self._init_ui()
    
    def _init_ui(self):        
        
        # Set background color to be transparent
        palette = QtGui.QPalette(self.palette())
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)
    
    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        
        #Overlay color is half-transparent white
        painter.fillRect(event.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 127)))
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        
        # Loop over dots in the 'wheel'
        # At any time, a single ellipse is colored bright
        # Other ellipses are darker
        for i in range(self._num_points):
            if  i == self.counter % self._num_points :  # Color this ellipse bright
                painter.setBrush(QtGui.QBrush(QtGui.QColor(229, 33, 33)))
            else:   # Color this ellipse dark
               painter.setBrush(QtGui.QBrush(QtGui.QColor(114, 15, 15)))
              
            # Draw the ellipse with the right color
            painter.drawEllipse(
                self.width()/2 + 30 * cos(2 * pi * i / self._num_points),
                self.height()/2 + 30 * sin(2 * pi * i / self._num_points),
                10, 10)

        painter.end()
    
    def showEvent(self, event):
        """ Starts an updating timer, called every 50 ms. """
        self.timer = self.startTimer(50)
        self.counter = 0
    
    def timerEvent(self, event):
        """ At every timer step, this method is called. """
        self.counter += 1
        self.update()       # Calls a paintEvent