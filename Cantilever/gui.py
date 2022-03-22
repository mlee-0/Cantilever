'''
Run this script to run the main script from a GUI.
'''


from queue import Queue
import sys
import threading

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QGridLayout, QHBoxLayout, QWidget, QPushButton, QCheckBox, QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QFrame

import main
import networks
from setup import Colors


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Queues used to communicate between threads.
        self.queue = Queue()
        self.queue_to_main = Queue()

        # Font objects.
        font_small = QFont()
        font_small.setPointSize(10)

        # Menu bar.
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        menu_help = menu_bar.addMenu("Help")
        menu_help.addAction("About...", self.on_about)

        # Central widget.
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(main_widget)
        
        self.sidebar = QWidget()
        layout_sidebar = QVBoxLayout(self.sidebar)
        layout_sidebar.setAlignment(Qt.AlignTop)
        layout_results = QVBoxLayout()
        main_layout.addWidget(self.sidebar)
        main_layout.addLayout(layout_results)
        
        # Train and test buttons.
        self.button_train = QPushButton("Train")
        self.button_test = QPushButton("Test")
        self.button_train.clicked.connect(self.on_start)
        self.button_test.clicked.connect(self.on_start)
        layout = QHBoxLayout()
        layout.addWidget(self.button_train)
        layout.addWidget(self.button_test)
        layout_sidebar.addLayout(layout)

        # Checkbox for toggling continuing training a previously saved model.
        self.checkbox_keep_training = QCheckBox("Keep Training Previous")
        self.checkbox_keep_training.setChecked(True)
        layout_sidebar.addWidget(self.checkbox_keep_training)

        # Settings.
        label_epochs = QLabel("Epochs:")
        self.value_epochs = QSpinBox()
        self.value_epochs.setMinimum(1)
        self.value_epochs.setMaximum(1_000_000)
        self.value_epochs.setSingleStep(10)
        self.value_epochs.setValue(100)
        self.value_epochs.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(label_epochs)
        layout.addWidget(self.value_epochs)
        layout_sidebar.addLayout(layout)
        
        label_learning = QLabel("Learning Rate:")
        self.value_learning = QDoubleSpinBox()
        self.value_learning.setMinimum(1e-10)
        self.value_learning.setMaximum(1.0)
        self.value_learning.setDecimals(5)
        self.value_learning.setSingleStep(1e-4)
        self.value_learning.setValue(0.00001)
        self.value_learning.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(label_learning)
        layout.addWidget(self.value_learning)
        layout_sidebar.addLayout(layout)

        label_batch = QLabel("Batch Size:")
        self.value_batch = QSpinBox()
        self.value_batch.setMinimum(1)
        self.value_batch.setValue(1)
        self.value_batch.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(label_batch)
        layout.addWidget(self.value_batch)
        layout_sidebar.addLayout(layout)

        label_model = QLabel("Model:")
        self.value_model = QComboBox()
        self.value_model.addItems(networks.networks.keys())
        layout = QHBoxLayout()
        layout.addWidget(label_model)
        layout.addWidget(self.value_model)
        layout_sidebar.addLayout(layout)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        layout_sidebar.addWidget(divider)

        label_dataset_size = QLabel("Dataset Size:")
        self.value_dataset_size = QSpinBox()
        self.value_dataset_size.setMinimum(1)
        self.value_dataset_size.setMaximum(100_000)
        self.value_dataset_size.setSingleStep(10)
        self.value_dataset_size.setValue(530)
        self.value_dataset_size.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(label_dataset_size)
        layout.addWidget(self.value_dataset_size)
        layout_sidebar.addLayout(layout)
        
        label_bins = QLabel("Bins:")
        self.value_bins = QSpinBox()
        self.value_bins.setMinimum(1)
        self.value_bins.setMaximum(1000)
        self.value_bins.setSingleStep(1)
        self.value_bins.setValue(10)
        self.value_bins.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(label_bins)
        layout.addWidget(self.value_bins)
        layout_sidebar.addLayout(layout)

        label_training_split = QLabel("Training Split:")
        self.value_training_split = QSpinBox()
        self.value_training_split.setMinimum(1)
        self.value_training_split.setMaximum(99)
        self.value_training_split.setSingleStep(5)
        self.value_training_split.setValue(80)
        self.value_training_split.setSuffix("%")
        self.value_training_split.setAlignment(Qt.AlignRight)
        self.value_training_split.valueChanged.connect(self.on_training_split_changed)
        layout = QHBoxLayout()
        layout.addWidget(label_training_split)
        layout.addWidget(self.value_training_split)
        layout_sidebar.addLayout(layout)

        self.label_training_dataset_size = QLabel()
        self.label_training_dataset_size.setFont(font_small)
        self.label_training_dataset_size.setAlignment(Qt.AlignCenter)
        layout_sidebar.addWidget(self.label_training_dataset_size)
        self.on_training_split_changed()

        # Progress bar and stop button.
        self.label_progress = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.on_stop)
        self.button_stop.setEnabled(False)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label_progress)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.button_stop)
        layout_results.addLayout(layout)

        # Plot area.
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout_results.addWidget(self.canvas)

        # Timer that checkes the queue for information from main thread.
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.setInterval(10)
    
    def on_start(self):
        """Start training or testing."""
        if self.sender() is self.button_train:
            train_model = self.checkbox_keep_training.isChecked()
        elif self.sender() is self.button_test:
            train_model = False

        self.sidebar.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.thread = threading.Thread(
            target=main.main,
            args=[self.value_epochs.value(), self.value_learning.value(), self.value_batch.value(), self.value_dataset_size.value(), self.value_bins.value(), self.value_training_split.value()/100, networks.networks[self.value_model.currentText()], train_model, self.queue, self.queue_to_main],
        )
        self.thread.start()
        self.timer.start()
    
    def on_stop(self):
        """Stop training after the current epoch has ended."""
        self.button_stop.setEnabled(False)
        self.queue_to_main.put(True)
    
    def on_training_split_changed(self):
        """Show a label displaying how many samples will be in the training dataset after splitting."""
        self.label_training_dataset_size.setText(
            f"{round((self.value_training_split.value()/100) * self.value_dataset_size.value())} in training dataset"
        )
    
    def on_about(self):
        """Show a window displaying the README file."""
        with open("README.md", 'r') as f:
            text = f.read()

        text_edit = QTextEdit(readOnly=True)
        text_edit.setMarkdown(text)

        window = QMainWindow(self)
        window.setWindowTitle("About")
        window.setCentralWidget(text_edit)
        window.show()

    def plot_loss(self, epochs, loss, previous_loss):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if previous_loss:
            axis.plot(range(epochs[0]), previous_loss, 'o', color=Colors.GRAY)
        axis.plot(epochs, loss, '-o', color=Colors.BLUE)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Loss")
        axis.grid(axis='y')
        self.canvas.draw()
    
    def check_queue(self):
        while not self.queue.empty():
            epoch, epochs, loss, previous_loss = self.queue.get()
            self.progress_bar.setValue(epoch)
            self.progress_bar.setMaximum(max(epochs))
            self.label_progress.setText(f"{epoch+1}/{max(epochs)+1}")
            self.plot_loss(epochs[:len(loss)], loss, previous_loss)
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