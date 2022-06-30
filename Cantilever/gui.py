"""
Run this script to run the main script as a GUI.
"""


import os
from queue import Queue
import sys
import threading
from typing import Tuple

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QGridLayout, QButtonGroup, QWidget, QScrollArea, QTabWidget, QPushButton, QRadioButton, QCheckBox, QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QFrame, QFileDialog

from helpers import Colors, FOLDER_ROOT, array_to_colormap
import main
import networks


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
        self.label_status = QLabel()
        self.status_bar.addWidget(self.label_status)

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
        
        self.sidebar = self._sidebar()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.sidebar)
        scroll_area.setFrameShape(QFrame.NoFrame)

        tabs = QTabWidget()
        tabs.addTab(self._plots(), "Training")
        tabs.addTab(self._evaluation(), "Testing")
        
        layout_results = QVBoxLayout()
        layout_results.addWidget(self._progress_bar())
        layout_results.addWidget(tabs)

        main_layout.addWidget(scroll_area)
        main_layout.addLayout(layout_results, stretch=1)
        
        # divider = QFrame()
        # divider.setFrameShape(QFrame.HLine)
        # layout_sidebar.addWidget(divider)

        # Timer that checkes the queue for information from main thread.
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.setInterval(100)
    
    def _sidebar(self) -> QWidget:
        """Return a widget containing fields for adjusting settings."""
        layout_sidebar = QVBoxLayout()
        layout_sidebar.setAlignment(Qt.AlignTop)
        widget = QWidget()
        widget.setLayout(layout_sidebar)

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
        self.checkbox_train_existing = QCheckBox("Train Existing")
        self.checkbox_train_existing.setChecked(True)
        self.label_filename_model = QLineEdit("model.pth")
        self.label_filename_model.setAlignment(Qt.AlignRight)
        # self.button_browse_model = QPushButton("model.pth")
        # self.button_browse_model.clicked.connect(self.open_dialog_model)
        layout = QHBoxLayout()
        layout.addWidget(self.checkbox_train_existing)
        layout.addWidget(self.label_filename_model)
        layout_sidebar.addLayout(layout)

        # Settings.
        self.value_epochs = QSpinBox()
        self.value_epochs.setRange(1, 1_000_000)
        self.value_epochs.setSingleStep(10)
        self.value_epochs.setValue(10)
        self.value_epochs.setAlignment(Qt.AlignRight)
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Epochs:"))
        layout.addWidget(self.value_epochs)
        layout_sidebar.addLayout(layout)
        
        self.value_learning_digit = QSpinBox()
        self.value_learning_digit.setRange(1, 9)
        self.value_learning_digit.setAlignment(Qt.AlignRight)
        self.value_learning_exponent = QSpinBox()
        self.value_learning_exponent.setRange(1, 10)
        self.value_learning_exponent.setValue(3)
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

        self.value_train_split = QSpinBox()
        self.value_train_split.setRange(1, 99)
        self.value_train_split.setValue(80)
        self.value_train_split.setAlignment(Qt.AlignRight)
        self.value_train_split.setToolTip("Training split")
        self.value_validate_split = QSpinBox()
        self.value_validate_split.setRange(1, 99)
        self.value_validate_split.setValue(10)
        self.value_validate_split.setAlignment(Qt.AlignRight)
        self.value_train_split.setToolTip("Validation split")
        self.value_test_split = QSpinBox()
        self.value_test_split.setRange(1, 99)
        self.value_test_split.setValue(10)
        self.value_test_split.setAlignment(Qt.AlignRight)
        self.value_train_split.setToolTip("Testing split")
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(QLabel("Split:"))
        layout.addStretch(1)
        layout.addWidget(self.value_train_split)
        layout.addWidget(self.value_validate_split)
        layout.addWidget(self.value_test_split)
        layout_sidebar.addLayout(layout)

        self.value_model = QComboBox()
        self.value_model.addItems(networks.networks.keys())
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Model:"))
        layout.addWidget(self.value_model)
        layout_sidebar.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Dataset:"))
        layout.addStretch(1)
        self.buttons_dataset = QButtonGroup()
        buttons = {2: QRadioButton("2D"), 3: QRadioButton("3D")}
        for id in buttons:
            self.buttons_dataset.addButton(buttons[id], id=id)
            layout.addWidget(buttons[id])
        buttons[2].setChecked(True)
        layout_sidebar.addLayout(layout)

        # Settings for selecting a subset.
        fields_subset = self._fields_subset()
        layout_sidebar.addWidget(fields_subset)
        self.update_button_browse_subset(self.checkbox_use_subset.isChecked())

        return widget

    def _fields_subset(self) -> QWidget:
        """Return a widget containing fields for selecting subset files."""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout()
        self.checkbox_use_subset = QCheckBox("Use Subset")
        self.checkbox_use_subset.setChecked(False)
        self.checkbox_use_subset.stateChanged.connect(self.update_button_browse_subset)
        layout.addWidget(self.checkbox_use_subset)
        self.button_browse_subset = QPushButton()
        self.button_browse_subset.clicked.connect(self.open_dialog_subset)
        layout.addWidget(self.button_browse_subset)

        main_layout.addLayout(layout)

        return widget

    def _progress_bar(self) -> QWidget:
        """Return a widget containing a progress bar."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self.label_progress = QLabel()
        self.label_progress_secondary = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.on_stop)
        self.button_stop.setEnabled(False)
        self.button_stop.setToolTip("Stop after current epoch.")
        
        layout.addWidget(self.label_progress)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.button_stop)

        return widget

    def _plots(self) -> QWidget:
        """Return a widget with plots."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0 ,0)

        self.figure_loss = Figure()
        self.canvas_loss = FigureCanvasQTAgg(self.figure_loss)
        layout.addWidget(self.canvas_loss)

        self.figure_metrics = Figure()
        self.canvas_metrics = FigureCanvasQTAgg(self.figure_metrics)
        # layout.addWidget(self.canvas_metrics)

        # scroll_area = QScrollArea()
        # scroll_area.setWidget(widget)

        return widget
    
    def _evaluation(self) -> QWidget:
        """Return a widget containing testing results."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignTop)

        # Arrays storing the test inputs, outputs, and labels.
        # self.test_inputs = np.empty((0, 0))
        self.test_outputs = np.empty((0, 0))
        self.test_labels = np.empty((0, 0))
        # Maximum value of dataset, used to determine range for colormap images.
        self.test_max_value = None

        # Labels that show an input, output, and label.
        layout_samples = QGridLayout()
        self.label_test_input = QLabel()
        self.label_test_output = QLabel()
        self.label_test_label = QLabel()
        # layout_samples.addWidget(QLabel("Input"), 0, 0, alignment=Qt.AlignCenter)
        layout_samples.addWidget(QLabel("Output"), 0, 0, alignment=Qt.AlignCenter)
        layout_samples.addWidget(QLabel("Label"), 0, 1, alignment=Qt.AlignCenter)
        # layout_samples.addWidget(self.label_test_input, 1, 0, alignment=Qt.AlignCenter)
        layout_samples.addWidget(self.label_test_output, 1, 0, alignment=Qt.AlignCenter)
        layout_samples.addWidget(self.label_test_label, 1, 1, alignment=Qt.AlignCenter)
        layout.addLayout(layout_samples)

        # Controls for showing different samples.
        self.test_index = 0
        self.test_channel = 0
        self.value_test_index = QSpinBox()
        self.value_test_index.setRange(1, 1)
        self.value_test_index.valueChanged.connect(self.show_test_outputs)
        self.value_test_channel = QSpinBox()
        self.value_test_channel.setRange(1, 1)
        self.value_test_channel.valueChanged.connect(self.show_test_outputs)
        self.value_test_scaling = QDoubleSpinBox()
        self.value_test_scaling.setRange(0.1, 10.0)
        self.value_test_scaling.setValue(1.0)
        self.value_test_scaling.valueChanged.connect(self.show_test_outputs)
        layout_controls = QHBoxLayout()
        layout_controls.addStretch(1)
        layout_controls.addWidget(QLabel("Sample:"))
        layout_controls.addWidget(self.value_test_index)
        layout_controls.addWidget(QLabel("Channel:"))
        layout_controls.addWidget(self.value_test_channel)
        layout_controls.addWidget(QLabel("Size:"))
        layout_controls.addWidget(self.value_test_scaling)
        layout_controls.addStretch(1)
        layout.addLayout(layout_controls)

        self.show_test_outputs()

        # Label that shows evaluation metrics.
        self.label_metrics = QLabel()
        self.label_metrics.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_metrics)

        return widget

    def on_start(self):
        """Start training or testing."""
        if self.sender() is self.button_train:
            train_existing = self.checkbox_train_existing.isChecked()
            train, test = True, True
        elif self.sender() is self.button_test:
            train_existing = False
            train, test = False, True

        self.sidebar.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.button_stop.setEnabled(True)

        self.thread = threading.Thread(
            target=main.main,
            kwargs={
                "epoch_count": self.value_epochs.value(),
                "learning_rate": self.value_learning_digit.value() * 10 ** -self.value_learning_exponent.value(),
                "batch_sizes": (self.value_batch.value(), 128, 128),
                "Model": networks.networks[self.value_model.currentText()],
                "dataset_id": self.buttons_dataset.checkedId(),
                "training_split": (
                    self.value_train_split.value() / 100,
                    self.value_validate_split.value() / 100,
                    self.value_test_split.value() / 100,
                ),
                "filename_model": self.label_filename_model.text(),
                "filename_subset": self.button_browse_subset.text() if self.checkbox_use_subset.isChecked() else None,
                "train_existing": train_existing,
                "train": train,
                "test": test,
                "evaluate": True,
                "queue": self.queue,
                "queue_to_main": self.queue_to_main,
            },
        )
        self.thread.start()
        self.timer.start()
    
    def on_stop(self):
        """Stop training after the current epoch has ended."""
        self.button_stop.setEnabled(False)
        self.queue_to_main.put(True)
    
    def update_button_browse_subset(self, state):
        self.button_browse_subset.setEnabled(state)
        if not state:
            self.button_browse_subset.setText("Browse...")

    def update_status_bar(self, text):
        """Display text in the status bar."""
        if text.isprintable():
            self.label_status.setText(text)

    def toggle_status_bar(self):
        """Toggle visibility of status bar."""
        self.status_bar.setVisible(self.action_toggle_status_bar.isChecked())
    
    def open_dialog_model(self):
        """Show a file dialog to choose an existing model file or specify a new model file name."""
        dialog = QFileDialog(self, directory=FOLDER_ROOT, filter="(*.pth)")
        dialog.setFileMode(QFileDialog.AnyFile)
        if dialog.exec_():
            files = dialog.selectedFiles()
            file = os.path.basename(files[0])
            if file:
                self.button_browse_model.setText(file)

    def open_dialog_subset(self):
        """Show a file dialog to choose an existing subset file."""
        filename, _ = QFileDialog.getOpenFileName(self, directory=FOLDER_ROOT, filter="(*.txt)")

        filename = os.path.basename(filename)
        if filename:
            self.button_browse_subset.setText(filename)

    def show_about(self):
        """Show a window displaying the README file."""
        with open("README.md", "r") as f:
            text = f.read()

        text_edit = QTextEdit(readOnly=True)
        text_edit.setMarkdown(text)

        window = QMainWindow(self)
        window.setWindowTitle("About")
        window.setCentralWidget(text_edit)
        window.show()

    def plot_loss(self, epochs, training_loss, previous_training_loss, validation_loss, previous_validation_loss):
        self.figure_loss.clear()
        axis = self.figure_loss.add_subplot(1, 1, 1)  # Number of rows, number of columns, index
        
        # Plot previous losses.
        if previous_training_loss and not self.action_toggle_loss.isChecked():
            axis.plot(
                range(epochs[0] - len(previous_training_loss), epochs[0]),
                previous_training_loss,
                ".:", color=Colors.GRAY_LIGHT
            )
        if previous_validation_loss and not self.action_toggle_loss.isChecked():
            axis.plot(
                range(epochs[0] - len(previous_validation_loss), epochs[0]),
                previous_validation_loss,
                ".-", color=Colors.GRAY_LIGHT
            )
        
        # Plot current losses.
        if training_loss:
            axis.plot(epochs[:len(training_loss)], training_loss, ".:", color=Colors.ORANGE, label="Training")
            axis.annotate(f"{training_loss[-1]:,.0f}", (epochs[len(training_loss)-1], training_loss[-1]), color=Colors.ORANGE, fontsize=10)
            axis.legend()
        if validation_loss:
            axis.plot(epochs[:len(validation_loss)], validation_loss, ".-", color=Colors.BLUE, label="Validation")
            axis.annotate(f"{validation_loss[-1]:,.0f}", (epochs[len(validation_loss)-1], validation_loss[-1]), color=Colors.BLUE, fontsize=10)
            axis.legend()
        
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Loss")
        axis.grid(axis="y")
        self.canvas_loss.draw()
    
    def plot_metrics(self):
        NUMBER_COLUMNS = 2
        self.figure_metrics.clear()
        # axis = self.figure_metrics.add_subplot(, NUMBER_COLUMNS, 1)

        self.canvas_metrics.draw()
    
    def show_test_outputs(self, value=1):
        """Display images of the testing results."""
        
        if self.sender == self.value_test_index:
            self.test_index = value - 1
        elif self.sender == self.value_test_channel:
            self.test_channel = value - 1

        if 0 not in self.test_outputs.shape:
            # input = self.test_inputs[self.test_index, self.test_channel, ...]
            output = self.test_outputs[self.test_index, self.test_channel, ...]
            label = self.test_labels[self.test_index, self.test_channel, ...]
            output_image = QImage(
                array_to_colormap(output.T, self.test_max_value).astype(np.int16),
                output.shape[1], output.shape[0], QImage.Format_RGB16,
            )
            label_image = QImage(
                array_to_colormap(label.T, self.test_max_value).astype(np.int16),
                label.shape[1], label.shape[0], QImage.Format_RGB16,
            )

            scaling = self.value_test_scaling.value()
            if scaling != 1:
                output_image = output_image.scaled(int(scaling * output.shape[1]), int(scaling * output.shape[0]), Qt.IgnoreAspectRatio)
                label_image = label_image.scaled(int(scaling * label.shape[1]), int(scaling * label.shape[0]), Qt.IgnoreAspectRatio)

            self.label_test_output.setPixmap(QPixmap(output_image))
            self.label_test_label.setPixmap(QPixmap(label_image))

    def check_queue(self):
        while not self.queue.empty():
            info = self.queue.get()
            progress_epoch: Tuple[int, int] = info.get("progress_epoch", (0, 0))
            progress_batch: Tuple[int, int] = info.get("progress_batch", (0, 0))
            epochs = info.get("epochs", range(0))
            training_loss = info.get("training_loss", None)
            previous_training_loss = info.get("previous_training_loss", None)
            validation_loss = info.get("validation_loss", None)
            previous_validation_loss = info.get("previous_validation_loss", None)
            values_metrics = info.get("values_metrics", {})

            # self.test_inputs = info.get("test_inputs", self.test_inputs)
            self.test_outputs = info.get("test_outputs", self.test_outputs)
            self.test_labels = info.get("test_labels", self.test_labels)
            self.test_max_value = info.get("test_max_value", self.test_max_value)
            if 0 not in self.test_outputs.shape:
                self.value_test_index.setMaximum(self.test_outputs.shape[0])
                self.value_test_channel.setMaximum(self.test_outputs.shape[1])
                self.show_test_outputs()

            # Update the progress label.
            strings_progress = []
            strings_progress.append(f"Epoch {progress_epoch[0]}/{progress_epoch[1]}")
            strings_progress.append(f"Batch {progress_batch[0]}/{progress_batch[1]}")
            text_progress = "\n".join(strings_progress)
            self.label_progress.setText(text_progress)

            # Update the progress bar.
            progress_value = max(progress_epoch[0]-1, 0) * progress_batch[1] + progress_batch[0]
            progress_max = max(progress_epoch[1], 1) * progress_batch[1]
            self.progress_bar.setValue(progress_value)
            self.progress_bar.setMaximum(progress_max)

            # Update the metrics.
            if values_metrics:
                text = "\n".join(
                    [f"{key}: {value}" for key, value in values_metrics.items()]
                )
                self.label_metrics.setText(text)

            if training_loss or previous_training_loss or validation_loss or previous_validation_loss:
                self.plot_loss(epochs, training_loss, previous_training_loss, validation_loss, previous_validation_loss)
        # Thread has stopped.
        if not self.thread.is_alive():
            self.sidebar.setEnabled(True)
            self.button_stop.setEnabled(False)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.reset()
            self.checkbox_train_existing.setChecked(True)
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