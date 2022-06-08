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
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QGridLayout, QButtonGroup, QWidget, QScrollArea, QPushButton, QRadioButton, QCheckBox, QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QFrame, QFileDialog

import main
import networks
from setup import Colors, FOLDER_ROOT


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
        layout_results = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        main_layout.addLayout(layout_results, stretch=1)
        
        # divider = QFrame()
        # divider.setFrameShape(QFrame.HLine)
        # layout_sidebar.addWidget(divider)

        # Progress bar and stop button.
        self.label_progress = QLabel()
        self.label_progress_secondary = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.on_stop)
        self.button_stop.setEnabled(False)
        self.button_stop.setToolTip("Stop after current epoch.")
        layout = QHBoxLayout()
        layout.addWidget(self.label_progress)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.button_stop)
        layout_results.addLayout(layout)

        # Plot area.
        layout_results.addLayout(self._plots())

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
        self.value_epochs.setMinimum(1)
        self.value_epochs.setMaximum(1_000_000)
        self.value_epochs.setSingleStep(10)
        self.value_epochs.setValue(50)
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

        self.value_train_split = QSpinBox()
        self.value_train_split.setMinimum(1)
        self.value_train_split.setMaximum(99)
        self.value_train_split.setValue(80)
        self.value_train_split.setAlignment(Qt.AlignRight)
        self.value_train_split.setToolTip("Training split")
        self.value_validate_split = QSpinBox()
        self.value_validate_split.setMinimum(1)
        self.value_validate_split.setMaximum(99)
        self.value_validate_split.setValue(10)
        self.value_validate_split.setAlignment(Qt.AlignRight)
        self.value_train_split.setToolTip("Validation split")
        self.value_test_split = QSpinBox()
        self.value_test_split.setMinimum(1)
        self.value_test_split.setMaximum(99)
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

    def _plots(self) -> QVBoxLayout:
        """Return a layout with plots."""
        # widget = QTabWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0 ,0)

        self.figure_loss = Figure()
        self.canvas_loss = FigureCanvasQTAgg(self.figure_loss)
        layout.addWidget(self.canvas_loss)

        self.figure_metrics = Figure()
        self.canvas_metrics = FigureCanvasQTAgg(self.figure_metrics)
        # layout.addWidget(self.canvas_metrics)

        # scroll_area = QScrollArea()
        # scroll_area.setWidget(widget)

        return layout

    def on_start(self):
        """Start training or testing."""
        if self.sender() is self.button_train:
            train_model = self.checkbox_train_existing.isChecked()
            test_only = False
        elif self.sender() is self.button_test:
            train_model = False
            test_only = True

        self.sidebar.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.button_stop.setEnabled(True)

        self.thread = threading.Thread(
            target=main.main,
            kwargs={
                "epoch_count": self.value_epochs.value(),
                "learning_rate": self.value_learning_digit.value() * 10 ** -self.value_learning_exponent.value(),
                "batch_size": self.value_batch.value(),
                "Model": networks.networks[self.value_model.currentText()],
                "dataset_id": self.buttons_dataset.checkedId(),
                "training_split": (
                    self.value_train_split.value() / 100,
                    self.value_validate_split.value() / 100,
                    self.value_test_split.value() / 100,
                ),
                "filename_subset": self.button_browse_subset.text() if self.checkbox_use_subset.isChecked() else None,
                "filename_model": self.label_filename_model.text(),
                "train_existing": train_model,
                "test_only": test_only,
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
        with open("README.md", 'r') as f:
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
                '.:', color=Colors.GRAY_LIGHT
            )
        if previous_validation_loss and not self.action_toggle_loss.isChecked():
            axis.plot(
                range(epochs[0] - len(previous_validation_loss), epochs[0]),
                previous_validation_loss,
                '.-', color=Colors.GRAY_LIGHT
            )
        
        # Plot current losses.
        if training_loss:
            axis.plot(epochs[:len(training_loss)], training_loss, '.:', color=Colors.ORANGE, label="Training")
            axis.annotate(f"{training_loss[-1]:,.0f}", (epochs[len(training_loss)-1], training_loss[-1]), color=Colors.ORANGE, fontsize=10)
            axis.legend()
        if validation_loss:
            axis.plot(epochs[:len(validation_loss)], validation_loss, '.-', color=Colors.BLUE, label="Validation")
            axis.annotate(f"{validation_loss[-1]:,.0f}", (epochs[len(validation_loss)-1], validation_loss[-1]), color=Colors.BLUE, fontsize=10)
            axis.legend()
        
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Loss")
        axis.grid(axis='y')
        self.canvas_loss.draw()
    
    def plot_metrics(self):
        NUMBER_COLUMNS = 2
        self.figure_metrics.clear()
        # axis = self.figure_metrics.add_subplot(, NUMBER_COLUMNS, 1)

        self.canvas_metrics.draw()

    def check_queue(self):
        while not self.queue.empty():
            info = self.queue.get()
            progress_epoch: Tuple[int, int] = info["progress_epoch"]
            progress_batch: Tuple[int, int] = info["progress_batch"]
            epochs = info["epochs"]
            training_loss = info["training_loss"]
            previous_training_loss = info["previous_training_loss"]
            validation_loss = info["validation_loss"]
            previous_validation_loss = info["previous_validation_loss"]

            # Update the progress label.
            strings_progress = []
            if progress_epoch is not None:
                strings_progress.append(f"Epoch {progress_epoch[0]}/{progress_epoch[1]}")
            if progress_batch is not None:
                strings_progress.append(f"Batch {progress_batch[0]}/{progress_batch[1]}")
            text_progress = "\n".join(strings_progress)
            self.label_progress.setText(text_progress)

            # Update the progress bar.
            progress_value = (progress_epoch[0]-1) * progress_batch[1] + progress_batch[0]
            progress_max = progress_epoch[1] * progress_batch[1]
            self.progress_bar.setValue(progress_value)
            self.progress_bar.setMaximum(progress_max)

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