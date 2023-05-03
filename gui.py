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
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QMargins
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout, QButtonGroup, QWidget, QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QPushButton, QToolButton, QRadioButton, QCheckBox, QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QFrame, QGroupBox, QSplitter, QFileDialog

from preprocessing import array_to_colormap
import main
import networks


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Queues used to communicate between threads.
        self.queue = Queue()
        self.queue_to_main = Queue()

        # Font objects.
        self.font_small = QFont()
        self.font_small.setPointSize(10)

        # Margin objects.
        self.margins_small = QMargins(5, 5, 5, 5)

        # Menu bar.
        menu_bar = self.menuBar()
        menu_view = menu_bar.addMenu("View")
        menu_help = menu_bar.addMenu("Help")
        self.action_toggle_console = menu_view.addAction("Show Console", self.toggle_console)
        self.action_toggle_console.setCheckable(True)
        self.action_toggle_console.setChecked(False)
        # self.action_toggle_status_bar = menu_view.addAction("Show Status Bar", self.toggle_status_bar)
        # self.action_toggle_status_bar.setCheckable(True)
        # self.action_toggle_status_bar.setChecked(True)
        # menu_view.addSeparator()
        # self.action_toggle_loss = menu_view.addAction("Show Current Loss Only")
        # self.action_toggle_loss.setCheckable(True)
        # menu_help.addAction("About...", self.show_about)

        # # Status bar.
        # self.status_bar = self.statusBar()
        # self.label_status = QLabel()
        # self.status_bar.addWidget(self.label_status)

        # Automatically send console messages to the status bar.
        # https://stackoverflow.com/questions/44432276/print-out-python-console-output-to-qtextedit
        class Stream(QtCore.QObject):
            newText = QtCore.pyqtSignal(str)
            def write(self, text):
                self.newText.emit(str(text))
        sys.stdout = Stream(newText=self.update_console)

        # Central widget.
        main_widget = QWidget()
        main_layout = QGridLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        self.sidebar = self._sidebar()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.sidebar)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Console.
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(self.font_small)
        self.console.setVisible(False)

        tabs = QTabWidget()
        tabs.addTab(self._training_tab(), "Training")
        tabs.addTab(self._testing_tab(), "Testing")
        
        layout_results = QVBoxLayout()
        layout_results.addWidget(self._progress_bar())
        layout_results.addWidget(tabs)

        main_layout.addWidget(scroll_area, 0, 0)
        main_layout.addWidget(self.console, 1, 0)
        main_layout.addLayout(layout_results, 0, 1, 2, 1)
        main_layout.setRowStretch(0, 5)
        main_layout.setRowStretch(1, 0)
        main_layout.setColumnStretch(1, 1)
        
        # Timer that checkes the queue for information from main thread.
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.setInterval(100)
    
    def _sidebar(self) -> QWidget:
        """Return a widget containing fields for adjusting settings."""
        layout_sidebar = QVBoxLayout()
        layout_sidebar.setContentsMargins(0, 0, 0, 0)
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

        # Continuing training of previously saved model settings.
        box = QGroupBox("Model")
        layout_box = QFormLayout(box)
        layout_box.setContentsMargins(self.margins_small)
        layout_sidebar.addWidget(box)

        self.field_filename_model = QLineEdit("model.pth")
        self.button_browse_model = QToolButton()
        self.button_browse_model.setIcon(QtWidgets.QFileIconProvider().icon(QtWidgets.QFileIconProvider.File))
        self.button_browse_model.clicked.connect(self.open_dialog_model)
        layout = QHBoxLayout()
        layout.addWidget(self.field_filename_model)
        layout.addWidget(self.button_browse_model)
        layout_box.addRow("Filename:", layout)
        
        self.checkbox_train_existing = QCheckBox("Keep Training")
        self.checkbox_train_existing.setChecked(True)
        layout_box.addRow("", self.checkbox_train_existing)

        self.value_save_every = QSpinBox()
        self.value_save_every.setMinimum(1)
        self.value_save_every.setSuffix(" epochs")
        layout_box.addRow("Save Every:", self.value_save_every)

        # Training hyperparameter settings.
        box = QGroupBox("Training")
        layout_box = QFormLayout(box)
        layout_box.setContentsMargins(self.margins_small)
        layout_sidebar.addWidget(box)

        self.value_epochs = QSpinBox()
        self.value_epochs.setRange(1, 1_000_000)
        self.value_epochs.setSingleStep(10)
        self.value_epochs.setValue(10)
        layout_box.addRow("Epochs:", self.value_epochs)
        
        self.value_learning_digit = QSpinBox()
        self.value_learning_digit.setRange(1, 9)
        self.value_learning_exponent = QSpinBox()
        self.value_learning_exponent.setRange(1, 10)
        self.value_learning_exponent.setValue(3)
        self.value_learning_exponent.setPrefix("−")
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.value_learning_digit)
        layout.addWidget(QLabel("e"))
        layout.addWidget(self.value_learning_exponent)
        layout_box.addRow("Learning Rate:", layout)

        self.checkbox_decay_learning_rate = QCheckBox("Decay")
        self.checkbox_decay_learning_rate.setChecked(True)
        layout_box.addRow("", self.checkbox_decay_learning_rate)

        self.value_batch_train = QSpinBox()
        self.value_batch_train.setRange(1, 256)
        self.value_batch_train.setValue(32)
        self.value_batch_train.setToolTip("Training batch size")
        self.value_batch_validate = QSpinBox()
        self.value_batch_validate.setRange(1, 256)
        self.value_batch_validate.setValue(128)
        self.value_batch_validate.setToolTip("Validation batch size")
        self.value_batch_test = QSpinBox()
        self.value_batch_test.setRange(1, 256)
        self.value_batch_test.setValue(128)
        self.value_batch_test.setToolTip("Testing batch size")
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.value_batch_train)
        layout.addWidget(self.value_batch_validate)
        layout.addWidget(self.value_batch_test)
        layout_box.addRow("Batch Size:", layout)

        self.value_train_split = QSpinBox()
        self.value_train_split.setRange(1, 99)
        self.value_train_split.setValue(80)
        self.value_train_split.setToolTip("Training split")
        self.value_validate_split = QSpinBox()
        self.value_validate_split.setRange(1, 99)
        self.value_validate_split.setValue(10)
        self.value_train_split.setToolTip("Validation split")
        self.value_test_split = QSpinBox()
        self.value_test_split.setRange(1, 99)
        self.value_test_split.setValue(10)
        self.value_train_split.setToolTip("Testing split")
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.value_train_split)
        layout.addWidget(self.value_validate_split)
        layout.addWidget(self.value_test_split)
        layout_box.addRow("Split:", layout)

        self.checkbox_k_fold = QCheckBox("k-Fold")
        self.checkbox_k_fold.setChecked(False)
        layout_box.addRow("", self.checkbox_k_fold)

        self.value_model = QComboBox()
        self.value_model.addItems(networks.networks.keys())
        layout = QHBoxLayout()
        layout_box.addRow("Model:", self.value_model)

        # Dataset settings.
        box = QGroupBox("Dataset")
        layout_box = QFormLayout(box)
        layout_box.setContentsMargins(self.margins_small)
        layout_sidebar.addWidget(box)

        layout = QHBoxLayout()
        self.buttons_dataset = QButtonGroup()
        buttons = {2: QRadioButton("2D"), 3: QRadioButton("3D")}
        for id in buttons:
            self.buttons_dataset.addButton(buttons[id], id=id)
            layout.addWidget(buttons[id])
        buttons[2].setChecked(True)
        layout_box.addRow("Dataset:", layout)
        
        self.value_transformation = QDoubleSpinBox()
        self.value_transformation.setRange(0, 10)
        self.value_transformation.setValue(1)
        layout_box.addRow("Transform:", self.value_transformation)

        self.checkbox_use_subset = QCheckBox("Use Subset")
        self.checkbox_use_subset.setChecked(False)
        self.checkbox_use_subset.stateChanged.connect(self.update_button_browse_subset)
        self.button_browse_subset = QPushButton()
        self.button_browse_subset.clicked.connect(self.open_dialog_subset)
        layout_box.addRow("", self.checkbox_use_subset)
        layout_box.addRow("", self.button_browse_subset)
        self.update_button_browse_subset(self.checkbox_use_subset.isChecked())

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

    def _training_tab(self) -> QWidget:
        """Return a widget to be used as the training tab."""
        widget = QSplitter()
        widget.setContentsMargins(self.margins_small)
        widget.setOpaqueResize(False)

        self.figure_loss = Figure()
        self.canvas_loss = FigureCanvasQTAgg(self.figure_loss)
        widget.addWidget(self.canvas_loss)

        self.figure_metrics = Figure()
        self.canvas_metrics = FigureCanvasQTAgg(self.figure_metrics)
        # layout.addWidget(self.canvas_metrics)

        self.table_training = QTableWidget(0, 2)
        self.table_training.horizontalHeader().hide()
        self.table_training.verticalHeader().hide()
        self.table_training.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # Fill available horizontal space
        self.table_training.setFont(self.font_small)
        widget.addWidget(self.table_training)
        
        widget.setStretchFactor(0, 4)
        widget.setStretchFactor(1, 1)
        
        widget.setCollapsible(0, False)

        return widget
    
    def _testing_tab(self) -> QWidget:
        """Return a widget to be used as the testing tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margins_small)
        layout.setAlignment(Qt.AlignTop)

        layout.addWidget(self._test_images())
        self.show_test_outputs()

        # Label that shows evaluation metrics.
        self.table_metrics = QTableWidget(0, 2)
        self.table_metrics.horizontalHeader().hide()
        self.table_metrics.verticalHeader().hide()
        self.table_metrics.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # Fill available horizontal space
        layout.addWidget(self.table_metrics)

        layout.addStretch(1)

        return widget
    
    def _test_images(self) -> QWidget:
        """Return a widget displaying test results and controls for selecting samples and channels."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Arrays storing the test inputs, outputs, and labels.
        self.test_inputs = np.empty((0, 0))
        self.test_outputs = np.empty((0, 0))
        self.test_labels = np.empty((0, 0))
        # Maximum value of dataset, used to determine range for colormap images.
        self.test_max_value = None

        # Labels that show an input, output, and label.
        self.label_test_input = QLabel()
        self.label_test_output = QLabel()
        self.label_test_label = QLabel()
        # Controls for showing different samples.
        self.test_index = 0
        self.test_channel_input = 0
        self.test_channel_output = 0
        self.value_test_index = QSpinBox()
        self.value_test_index.setRange(1, 1)
        self.value_test_index.valueChanged.connect(self.show_test_outputs)
        self.value_test_scaling = QDoubleSpinBox()
        self.value_test_scaling.setRange(0.1, 10.0)
        self.value_test_scaling.setValue(1.0)
        self.value_test_scaling.valueChanged.connect(self.show_test_outputs)
        self.value_test_input_channel = QSpinBox()
        self.value_test_input_channel.setRange(1, 1)
        self.value_test_input_channel.valueChanged.connect(self.show_test_outputs)
        self.value_test_output_channel = QSpinBox()
        self.value_test_output_channel.setRange(1, 1)
        self.value_test_output_channel.valueChanged.connect(self.show_test_outputs)

        # Controls for selecting the sample and the image scaling.
        layout_sample = QHBoxLayout()
        layout_sample.addWidget(QLabel("Sample:"))
        layout_sample.addWidget(self.value_test_index)
        layout_sample.addWidget(QLabel("Size:"))
        layout_sample.addWidget(self.value_test_scaling)

        # Controls for selecting the input image channel.
        layout_input_channel = QHBoxLayout()
        layout_input_channel.addWidget(QLabel("Channel:"))
        layout_input_channel.addWidget(self.value_test_input_channel)
        
        # Controls for selecting the output image channel.
        layout_output_channel = QHBoxLayout()
        layout_output_channel.addWidget(QLabel("Channel:"))
        layout_output_channel.addWidget(self.value_test_output_channel)

        layout.addWidget(QLabel("Input"), 0, 0, alignment=Qt.AlignCenter)
        layout.addWidget(QLabel("Output"), 0, 1, alignment=Qt.AlignCenter)
        layout.addWidget(QLabel("Label"), 0, 2, alignment=Qt.AlignCenter)
        layout.addWidget(self.label_test_input, 1, 0, alignment=Qt.AlignCenter)
        layout.addWidget(self.label_test_output, 1, 1, alignment=Qt.AlignCenter)
        layout.addWidget(self.label_test_label, 1, 2, alignment=Qt.AlignCenter)
        layout.addLayout(layout_input_channel, 2, 0, alignment=Qt.AlignCenter)
        layout.addLayout(layout_output_channel, 2, 1, 1, 2, alignment=Qt.AlignCenter)
        layout.addLayout(layout_sample, 3, 0, 1, 3, alignment=Qt.AlignCenter)

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
        self.console.clear()
        self.progress_bar.setRange(0, 0)
        self.button_stop.setEnabled(True)
        self.table_training.setRowCount(0)
        self.table_training.clear()
        self.table_metrics.setRowCount(0)
        self.table_metrics.clear()

        self.thread = threading.Thread(
            target=main.main,
            kwargs={
                "filename_model": self.field_filename_model.text(),
                "train_existing": train_existing,
                "save_model_every": self.value_save_every.value(),

                "epoch_count": self.value_epochs.value(),
                "learning_rate": self.value_learning_digit.value() * 10 ** -self.value_learning_exponent.value(),
                "decay_learning_rate": self.checkbox_decay_learning_rate.isChecked(),
                "batch_sizes": (self.value_batch_train.value(), self.value_batch_validate.value(), self.value_batch_test.value()),
                "training_split": (
                    self.value_train_split.value() / 100,
                    self.value_validate_split.value() / 100,
                    self.value_test_split.value() / 100,
                ),
                "k_fold": self.checkbox_k_fold.isChecked(),
                "Model": networks.networks[self.value_model.currentText()],
                
                "dataset_id": self.buttons_dataset.checkedId(),
                "transformation_exponent": self.value_transformation.value(),
                "filename_subset": self.button_browse_subset.text() if self.checkbox_use_subset.isChecked() else None,
                
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

    def update_console(self, text):
        """Display text in the console box."""
        if text.isprintable():
            self.console.insertPlainText(text)
            self.console.insertPlainText("\n")

    def toggle_status_bar(self):
        """Toggle visibility of status bar."""
        self.status_bar.setVisible(self.action_toggle_status_bar.isChecked())
    
    def toggle_console(self):
        """Toggle visibility of console."""
        self.console.setVisible(self.action_toggle_console.isChecked())

    def open_dialog_model(self):
        """Show a file dialog to choose an existing model file or specify a new model file name."""
        dialog = QFileDialog(self, directory='Checkpoints', filter="(*.pth)")
        dialog.setFileMode(QFileDialog.AnyFile)
        if dialog.exec_():
            files = dialog.selectedFiles()
            file = os.path.basename(files[0])
            if file:
                self.field_filename_model.setText(file)

    def open_dialog_subset(self):
        """Show a file dialog to choose an existing subset file."""
        filename, _ = QFileDialog.getOpenFileName(self, directory='.', filter="(*.txt)")

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
        
        all_training_loss = [*previous_training_loss, *training_loss]
        all_validation_loss = [*previous_validation_loss, *validation_loss]
        if training_loss:
            axis.plot(range(1, len(all_training_loss)+1), all_training_loss, ".:", label="Training")
            axis.annotate(f"{training_loss[-1]:,.2f}", (epochs[len(training_loss)-1], training_loss[-1]), fontsize=10)
        if validation_loss:
            axis.plot(range(1, len(all_validation_loss)+1), all_validation_loss, ".-", label="Validation")
            axis.annotate(f"{validation_loss[-1]:,.2f}", (epochs[len(validation_loss)-1], validation_loss[-1]), fontsize=10)

        if previous_training_loss or previous_validation_loss:
            axis.vlines(epochs[0] - 0.5, 0, max(previous_training_loss + previous_validation_loss), colors=([0]*3,), label="Current session starts")
        
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

        if self.sender() == self.value_test_index:
            self.test_index = value - 1
        elif self.sender() == self.value_test_input_channel:
            self.test_channel_input = value - 1
        elif self.sender() == self.value_test_output_channel:
            self.test_channel_output = value - 1

        if 0 not in self.test_outputs.shape:
            input = self.test_inputs[self.test_index, self.test_channel_input, ...]
            output = self.test_outputs[self.test_index, self.test_channel_output, ...]
            label = self.test_labels[self.test_index, self.test_channel_output, ...]
            input_image = QImage(
                input.astype(np.uint8),
                input.shape[1], input.shape[0], QImage.Format_Grayscale8,
            )
            output_image = QImage(
                array_to_colormap(output.T, self.test_max_value).astype(np.uint16),
                output.shape[1], output.shape[0], QImage.Format_RGB16,
            )
            label_image = QImage(
                array_to_colormap(label.T, self.test_max_value).astype(np.uint16),
                label.shape[1], label.shape[0], QImage.Format_RGB16,
            )

            scaling = self.value_test_scaling.value()
            if scaling != 1:
                input_image = input_image.scaled(int(scaling * input.shape[1]), int(scaling * input.shape[0]), Qt.IgnoreAspectRatio)
                output_image = output_image.scaled(int(scaling * output.shape[1]), int(scaling * output.shape[0]), Qt.IgnoreAspectRatio)
                label_image = label_image.scaled(int(scaling * label.shape[1]), int(scaling * label.shape[0]), Qt.IgnoreAspectRatio)

            self.label_test_input.setPixmap(QPixmap(input_image))
            self.label_test_output.setPixmap(QPixmap(output_image))
            self.label_test_label.setPixmap(QPixmap(label_image))

    def check_queue(self):
        while not self.queue.empty():
            info = self.queue.get()
            progress_epoch: Tuple[int, int] = info.get("progress_epoch", (0, 0))
            progress_batch: Tuple[int, int] = info.get("progress_batch", (0, 0))
            epochs = info.get("epochs", range(0))
            training_loss = info.get("training_loss", [])
            previous_training_loss = info.get("previous_training_loss", [])
            validation_loss = info.get("validation_loss", [])
            previous_validation_loss = info.get("previous_validation_loss", [])
            info_training = info.get("info_training", {})
            info_metrics = info.get("info_metrics", {})

            self.test_inputs = info.get("test_inputs", self.test_inputs)
            self.test_outputs = info.get("test_outputs", self.test_outputs)
            self.test_labels = info.get("test_labels", self.test_labels)
            self.test_max_value = info.get("test_max_value", self.test_max_value)
            if 0 not in self.test_outputs.shape:
                self.value_test_index.setSuffix(f"/{self.test_outputs.shape[0]}")
                self.value_test_index.setMaximum(self.test_outputs.shape[0])
                self.value_test_input_channel.setSuffix(f"/{self.test_inputs.shape[1]}")
                self.value_test_input_channel.setMaximum(self.test_inputs.shape[1])
                self.value_test_output_channel.setSuffix(f"/{self.test_outputs.shape[1]}")
                self.value_test_output_channel.setMaximum(self.test_outputs.shape[1])
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

            # Update training information.
            if info_training:
                for key, value in info_training.items():
                    items = self.table_training.findItems(key, Qt.MatchExactly)
                    if items:
                        row = self.table_training.row(items[0])
                    else:
                        row = self.table_training.rowCount()
                        self.table_training.insertRow(row)
                    self.table_training.setItem(row, 0, QTableWidgetItem(key))
                    self.table_training.setItem(row, 1, QTableWidgetItem(str(value)))

            # Update the metrics.
            if info_metrics:
                for metric, value in info_metrics.items():
                    items = self.table_metrics.findItems(metric, Qt.MatchExactly)
                    if items:
                        row = self.table_metrics.row(items[0])
                    else:
                        row = self.table_metrics.rowCount()
                        self.table_metrics.insertRow(row)
                    self.table_metrics.setItem(row, 0, QTableWidgetItem(metric))
                    self.table_metrics.setItem(row, 1, QTableWidgetItem(str(value)))
                self.table_metrics.resizeRowsToContents()
                self.table_metrics.resizeColumnsToContents()

            if training_loss or previous_training_loss or validation_loss or previous_validation_loss:
                all_training_loss = [*previous_training_loss, *training_loss]
                all_validation_loss = [*previous_validation_loss, *validation_loss]
                all_epochs = [*range(1, epochs[0]), *epochs[:len(training_loss)]]

                epochs = all_epochs
                loss = [all_training_loss, all_validation_loss]
                labels = ("Training", "Validation")
                start_epoch = epochs[0] if previous_training_loss else None

                self.figure_loss.clear()
                axis = self.figure_loss.add_subplot(1, 1, 1)  # Number of rows, number of columns, index
                
                # markers = (".:", ".-")

                # Plot each set of loss values.
                for i, loss_i in enumerate(loss):
                    if not len(loss_i):
                        continue
                    axis.semilogy(epochs[:len(loss_i)], loss_i, ".-", label=labels[i])
                    axis.annotate(f"{loss_i[-1]:,.2f}", (epochs[-1 - (len(epochs)-len(loss_i))], loss_i[-1]), fontsize=10)
                
                # Plot a vertical line indicating when the current training session began.
                if start_epoch:
                    axis.vlines(start_epoch - 0.5, 0, max([max(_) for _ in loss]), colors=([0.5]*3,), label="Current session starts")
                
                axis.legend()
                axis.set_xlabel("Epochs")
                axis.set_ylabel("Loss")
                axis.grid(axis="y")
                self.canvas_loss.draw()
                
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