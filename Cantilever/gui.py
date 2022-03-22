'''
Run this script to run the main script from a GUI.
'''


from queue import Queue
import sys
import threading

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QGridLayout, QHBoxLayout, QWidget, QPushButton, QCheckBox, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar
from PyQt5.QtCore import Qt, QTimer

import main
import networks


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.queue = Queue()
        self.queue_to_main = Queue()

        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        self.sidebar = QWidget()
        layout_sidebar = QVBoxLayout(self.sidebar)
        layout_sidebar.setContentsMargins(0, 0, 0, 0)
        layout_sidebar.setAlignment(Qt.AlignTop)
        layout.addWidget(self.sidebar)
        layout_results = QVBoxLayout()
        layout.addLayout(layout_results)

        self.label_progress = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.on_stop)
        self.button_stop.setEnabled(False)
        layout_progress = QHBoxLayout()
        layout_progress.addWidget(self.label_progress)
        layout_progress.addWidget(self.progress_bar)
        layout_progress.addWidget(self.button_stop)
        layout_results.addLayout(layout_progress)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout_results.addWidget(self.canvas)

        self.button_train = QPushButton("Train")
        self.button_test = QPushButton("Test")
        self.button_train.clicked.connect(self.on_start)
        self.button_test.clicked.connect(self.on_start)
        layout_buttons = QHBoxLayout()
        layout_buttons.addWidget(self.button_train)
        layout_buttons.addWidget(self.button_test)
        layout_sidebar.addLayout(layout_buttons)

        self.checkbox_continue_train = QCheckBox("Continue Training")
        self.checkbox_continue_train.setChecked(True)
        layout_sidebar.addWidget(self.checkbox_continue_train)

        label_epochs = QLabel("Epochs:")
        self.text_epochs = QSpinBox()
        self.text_epochs.setMinimum(1)
        self.text_epochs.setMaximum(1_000_000)
        self.text_epochs.setSingleStep(10)
        self.text_epochs.setValue(100)
        self.text_epochs.setAlignment(Qt.AlignRight)
        layout_epochs = QHBoxLayout()
        layout_epochs.addWidget(label_epochs)
        layout_epochs.addWidget(self.text_epochs)
        layout_sidebar.addLayout(layout_epochs)
        
        label_learning = QLabel("Learning Rate:")
        self.text_learning = QDoubleSpinBox()
        self.text_learning.setMinimum(1e-10)
        self.text_learning.setMaximum(1.0)
        self.text_learning.setDecimals(5)
        self.text_learning.setSingleStep(1e-4)
        self.text_learning.setValue(0.00001)
        self.text_learning.setAlignment(Qt.AlignRight)
        layout_learning = QHBoxLayout()
        layout_learning.addWidget(label_learning)
        layout_learning.addWidget(self.text_learning)
        layout_sidebar.addLayout(layout_learning)

        label_batch = QLabel("Batch Size:")
        self.text_batch = QSpinBox()
        self.text_batch.setMinimum(1)
        self.text_batch.setValue(1)
        self.text_batch.setAlignment(Qt.AlignRight)
        layout_batch = QHBoxLayout()
        layout_batch.addWidget(label_batch)
        layout_batch.addWidget(self.text_batch)
        layout_sidebar.addLayout(layout_batch)

        label_model = QLabel("Model:")
        self.text_model = QComboBox()
        self.text_model.addItems(networks.networks.keys())
        layout_model = QHBoxLayout()
        layout_model.addWidget(label_model)
        layout_model.addWidget(self.text_model)
        layout_sidebar.addLayout(layout_model)

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.setInterval(10)
    
    def on_start(self):
        if self.sender() is self.button_train:
            train_model = self.checkbox_continue_train.isChecked()
        elif self.sender() is self.button_test:
            train_model = False

        self.sidebar.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.thread = threading.Thread(
            target=main.main,
            args=[self.text_epochs.value(), self.text_learning.value(), self.text_batch.value(), networks.networks[self.text_model.currentText()], train_model, self.queue, self.queue_to_main],
        )
        self.thread.start()
        self.timer.start()
    
    def on_stop(self):
        self.button_stop.setEnabled(False)
        self.queue_to_main.put(True)
    
    def plot_loss(self, epochs, loss):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.plot(epochs[:len(loss)], loss, '-o')
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Loss")
        axis.grid(axis='y')
        self.canvas.draw()
    
    def check_queue(self):
        while not self.queue.empty():
            epoch, epochs, losses = self.queue.get()
            self.progress_bar.setValue(epoch)
            self.progress_bar.setMaximum(max(epochs))
            self.label_progress.setText(f"{epoch}/{max(epochs)}")
            self.plot_loss(epochs, losses)
        # Thread has stopped.
        if not self.thread.is_alive():
            self.sidebar.setEnabled(True)
            self.button_stop.setEnabled(False)
            self.progress_bar.reset()


if __name__ == "__main__":
    application = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Cantilever")
    window.show()
    sys.exit(application.exec_())