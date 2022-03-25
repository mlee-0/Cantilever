'''
Run this script to run the main script as a GUI.
'''


from queue import Queue
import sys
import threading

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QGridLayout, QHBoxLayout, QWidget, QPushButton, QCheckBox, QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QFrame

import main
import networks
from setup import split_training_validation, Colors


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
        menu_view = menu_bar.addMenu("View")
        menu_help = menu_bar.addMenu("Help")
        self.action_toggle_status_bar = menu_view.addAction("Show Status Bar", self.toggle_status_bar)
        self.action_toggle_status_bar.setCheckable(True)
        self.action_toggle_status_bar.setChecked(True)
        menu_view.addSeparator()
        self.action_toggle_loss = menu_view.addAction("Show Current Loss Only")
        self.action_toggle_loss.setCheckable(True)
        menu_help.addAction("About...", self.show_about)

        # Status bar.
        self.status_bar = self.statusBar()

        # Automatically send console messages to the status bar.
        # https://stackoverflow.com/questions/44432276/print-out-python-console-output-to-qtextedit
        class Stream(QtCore.QObject):
            newText = QtCore.pyqtSignal(str)
            def write(self, text):
                self.newText.emit(str(text))
        sys.stdout = Stream(newText=self.update_status_bar)

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
        self.value_epochs = QSpinBox()
        self.value_epochs.setMinimum(1)
        self.value_epochs.setMaximum(1_000_000)
        self.value_epochs.setSingleStep(10)
        self.value_epochs.setValue(100)
        self.value_epochs.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Epochs:"))
        layout.addWidget(self.value_epochs)
        layout_sidebar.addLayout(layout)
        
        self.value_learning_digit = QSpinBox()
        self.value_learning_digit.setMinimum(1)
        self.value_learning_digit.setMaximum(9)
        self.value_learning_digit.setAlignment(Qt.AlignRight)
        self.value_learning_exponent = QSpinBox()
        self.value_learning_exponent.setMinimum(1)
        self.value_learning_exponent.setMaximum(10)
        self.value_learning_exponent.setValue(7)
        self.value_learning_exponent.setPrefix("-")
        self.value_learning_exponent.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(QLabel("Learning Rate:"))
        layout.addStretch(1)
        layout.addWidget(self.value_learning_digit)
        layout.addWidget(QLabel("e"))
        layout.addWidget(self.value_learning_exponent)
        layout_sidebar.addLayout(layout)

        self.value_batch = QSpinBox()
        self.value_batch.setMinimum(1)
        self.value_batch.setValue(1)
        self.value_batch.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Batch Size:"))
        layout.addWidget(self.value_batch)
        layout_sidebar.addLayout(layout)

        self.value_model = QComboBox()
        self.value_model.addItems(networks.networks.keys())
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Model:"))
        layout.addWidget(self.value_model)
        layout_sidebar.addLayout(layout)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        layout_sidebar.addWidget(divider)

        self.value_dataset_size = QSpinBox()
        self.value_dataset_size.setMinimum(0)
        self.value_dataset_size.setMaximum(999_999)
        self.value_dataset_size.setSingleStep(10)
        self.value_dataset_size.setValue(10_000)
        self.value_dataset_size.setAlignment(Qt.AlignRight)
        self.value_dataset_size.valueChanged.connect(self.on_training_split_changed)
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Dataset Size:"))
        layout.addWidget(self.value_dataset_size)
        layout_sidebar.addLayout(layout)
        
        self.value_bins = QSpinBox()
        self.value_bins.setMinimum(1)
        self.value_bins.setMaximum(1000)
        self.value_bins.setSingleStep(1)
        self.value_bins.setValue(1)
        self.value_bins.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Bins:"))
        layout.addWidget(self.value_bins)
        layout_sidebar.addLayout(layout)

        self.value_nonuniformity = QDoubleSpinBox()
        self.value_nonuniformity.setMinimum(0.1)
        self.value_nonuniformity.setMaximum(100_000)
        self.value_nonuniformity.setDecimals(1)
        self.value_nonuniformity.setSingleStep(0.1)
        self.value_nonuniformity.setValue(1.0)
        self.value_nonuniformity.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Nonuniformity:"))
        layout.addWidget(self.value_nonuniformity)
        layout_sidebar.addLayout(layout)

        self.value_training_split = QSpinBox()
        self.value_training_split.setMinimum(1)
        self.value_training_split.setMaximum(99)
        self.value_training_split.setSingleStep(5)
        self.value_training_split.setValue(80)
        self.value_training_split.setSuffix("%")
        self.value_training_split.setAlignment(Qt.AlignRight)
        self.value_training_split.valueChanged.connect(self.on_training_split_changed)
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Training Split:"))
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
            test_only = False
        elif self.sender() is self.button_test:
            train_model = False
            test_only = True

        self.sidebar.setEnabled(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.button_stop.setEnabled(True)
        self.thread = threading.Thread(
            target=main.main,
            args=[
                self.value_epochs.value(),
                self.value_learning_digit.value() * 10 ** -self.value_learning_exponent.value(),
                self.value_batch.value(),
                self.value_dataset_size.value(),
                self.value_bins.value(),
                self.value_nonuniformity.value(),
                self.value_training_split.value()/100,
                networks.networks[self.value_model.currentText()],
                train_model, test_only, self.queue, self.queue_to_main
            ],
        )
        self.thread.start()
        self.timer.start()
    
    def on_stop(self):
        """Stop training after the current epoch has ended."""
        self.button_stop.setEnabled(False)
        self.queue_to_main.put(True)
    
    def update_status_bar(self, text):
        """Display text in the status bar."""
        if text.isprintable():
            self.status_bar.showMessage(text)

    def on_training_split_changed(self):
        """Show a label displaying how many samples will be in the training dataset after splitting."""
        dataset_size = self.value_dataset_size.value()
        training_size, validation_size = split_training_validation(dataset_size, self.value_training_split.value()/100)
        self.label_training_dataset_size.setText(
            f"{training_size} training / {validation_size} validation" if dataset_size > 0 else "Using all samples found"
        )
    
    def toggle_status_bar(self):
        """Toggle visibility of status bar."""
        self.status_bar.setVisible(self.action_toggle_status_bar.isChecked())
    
    def show_about(self):
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
        if previous_loss and not self.action_toggle_loss.isChecked():
            axis.plot(range(epochs[0]), previous_loss, 'o', color=Colors.GRAY)
        axis.plot(epochs, loss, '-o', color=Colors.BLUE)
        axis.annotate(f"{loss[-1].item():,.0f}", (epochs[-1], loss[-1]), color=Colors.BLUE, fontsize=10)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Loss")
        axis.grid(axis='y')
        self.canvas.draw()
    
    def check_queue(self):
        while not self.queue.empty():
            progress, epochs, loss, previous_loss = self.queue.get()
            self.progress_bar.setValue(progress[0])
            self.progress_bar.setMaximum(progress[1])
            self.label_progress.setText(f"{progress[0]}/{progress[1]}")
            if epochs is not None and loss is not None:
                self.plot_loss(epochs[:len(loss)], loss, previous_loss)
        # Thread has stopped.
        if not self.thread.is_alive():
            self.sidebar.setEnabled(True)
            self.button_stop.setEnabled(False)
            self.progress_bar.reset()
            self.checkbox_keep_training.setChecked(True)
            self.timer.stop()
    
    def closeEvent(self, event):
        """Base class method that closes the application."""
        # Return to default.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


if __name__ == "__main__":
    application = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Cantilever")
    window.show()
    sys.exit(application.exec_())