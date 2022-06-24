"""
Run this script to train and test the model.
"""


import gc
import glob
import os
from queue import Queue
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader

from datasets import *
from helpers import *
import metrics
from networks import *


# Free memory between subsequent runs.
if GOOGLE_COLAB:
    gc.collect()


class CantileverDataset(Dataset):
    """
    Dataset that contains 4D input images and label images. Generates input images and loads a .pickle file of pre-generated label images.

    Input images have shape (batch, channel, height, length).
    Label images have shape (batch, channel, height, length).
    """

    def __init__(self, samples: pd.DataFrame, is_3d: bool):
        self.number_samples = len(samples)

        if is_3d:
            folder_labels = os.path.join(FOLDER_ROOT, "Labels 3D")
            filename_labels = "labels_3d_to_50k.pickle"
        else:
            folder_labels = os.path.join(FOLDER_ROOT, "Labels 2D")
            filename_labels = "labels_2d_to_50k.pickle"
        
        # Load previously generated label images.
        self.labels = read_pickle(os.path.join(folder_labels, filename_labels))
        print(f"Label images take up {self.labels.nbytes/1e6:,.2f} MB.")

        # The maximum value found in the entire dataset.
        self.max_value = np.max(self.labels)

        # Determine an exponent to transform the data.
        self.transformation_exponent = optimize_transformation_exponent(self.labels, initial_guess=1/2.0, bounds=(1/3, 1))
        # Apply a transformation to the label values.
        self.labels = self.transform(self.labels, inverse=False)
        
        # Create input images.
        self.inputs = generate_input_images(samples, is_3d=is_3d)
        print(f"Input images take up {self.inputs.nbytes/1e6:,.2f} MB.")

        # Numerical inputs, scaled to [0, 1].
        self.loads = (samples[load.name] - load.low) / (load.high - load.low)

        # Number of channels in input and label images.
        self.input_channels = self.inputs.shape[1]
        self.output_channels = self.labels.shape[1]

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        """Return input data (tuple of image, list of numerical data) and label images."""
        # Return copies of arrays so that arrays are not modified.
        return (
            (np.copy(self.inputs[index, ...]), self.loads[index]),
            np.copy(self.labels[index, ...]),
        )
    
    def transform(self, y: np.ndarray, inverse=False) -> np.ndarray:
        """Raise the given data to an exponent, or the inverse of the exponent."""
        if not inverse:
            return y ** self.transformation_exponent
        else:
            return y ** (1 / self.transformation_exponent)

class CantileverDataset3d(Dataset):
    """
    Dataset that contains 5D input images and label images for use with 3D convolution. Generates input images and loads a .pickle file of pre-generated label images.

    Input images have shape (batch, channel, height, length, width).
    Label images have shape (batch, channel=1, height, length, width).
    """
    def __init__(self, samples: pd.DataFrame):
        self.number_samples = len(samples)

        folder_labels = os.path.join(FOLDER_ROOT, "Labels 3D")
        
        # Load previously generated label images.
        self.labels = read_pickle(os.path.join(folder_labels, "labels_3d_to_50k.pickle"))
        # Transpose dimensions for shape: (samples, 1, height (Y), length (X), width (Z)).
        self.labels = np.expand_dims(self.labels, axis=1).transpose((0, 1, 3, 4, 2))
        print(f"Label images take up {self.labels.nbytes/1e6:,.2f} MB.")

        # The maximum value found in the entire dataset.
        self.max_value = np.max(self.labels)

        # Apply a transformation to the label values.
        self.labels = self.transform(self.labels, inverse=False)
        
        # Create input images.
        self.inputs = generate_input_images_3d(samples)
        print(f"Input images take up {self.inputs.nbytes/1e6:,.2f} MB.")

        # Numerical inputs, scaled to [0, 1].
        self.loads = (samples[load.name] - load.low) / (load.high - load.low)

        # Number of channels in input and label images.
        self.input_channels = self.inputs.shape[1]
        self.output_channels = self.labels.shape[1]

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        """Return input data (tuple of image, list of numerical data) and label images."""
        # Return copies of arrays so that arrays are not modified.
        return (
            (np.copy(self.inputs[index, ...]), self.loads[index]),
            np.copy(self.labels[index, ...]),
        )
    
    @staticmethod
    def transform(y: np.ndarray, inverse=False) -> np.ndarray:
        if not inverse:
            return y ** (1/2)
        else:
            return y ** 2

class CantileverEncoderDataset(Dataset):
    """Dataset that contains input images and label images."""
    def __init__(self, samples: pd.DataFrame, is_3d: bool):
        self.number_samples = len(samples)

        folder_labels = os.path.join(FOLDER_ROOT, "Labels 3D" if is_3d else "Labels")
        
        # Load previously generated label images.
        files = glob.glob(os.path.join(folder_labels, "*.pickle"))
        files.sort()
        if files:
            file = files[0]
            self.labels = read_pickle(file)
        # Create label images and save them as a pickle file.
        else:
            self.labels = generate_label_images(samples, folder_labels, is_3d=is_3d)
            file = os.path.join(folder_labels, f"labels.pickle")
            write_pickle(self.labels, file)
        print(f"Label images take up {self.labels.nbytes/1e6:,.2f} MB.")

        # The maximum value found in the entire dataset.
        self.max_value = np.max(self.labels)

        # Apply a transformation to the label values.
        self.labels = self.transform(self.labels, inverse=False)
        
        # Create input images.
        self.inputs = generate_input_images(samples, is_3d=is_3d)
        print(f"Input images take up {self.inputs.nbytes/1e6:,.2f} MB.")

        # Numerical inputs, scaled to [0, 1].
        self.loads = (samples[load.name] - load.low) / (load.high - load.low)

        # Number of channels in input and label images.
        self.input_channels = self.inputs.shape[1]
        self.output_channels = self.labels.shape[1]

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        """Return input data (tuple of image, list of numerical data) and label images."""
        # Return copies of arrays so that arrays are not modified.
        return (
            (np.copy(self.inputs[index, ...]), self.loads[index]),
            np.copy(self.labels[index, ...]),
        )

def save_model(filepath: str, **kwargs) -> None:
    """Save model parameters to a file."""
    torch.save(kwargs, filepath)
    print(f"Saved model parameters to {filepath}.")

def load_model(filepath: str, device: str) -> dict:
    try:
        checkpoint = torch.load(filepath, map_location=device)
    except FileNotFoundError:
        print(f"{filepath} not found.")
    else:
        print(f"Loaded model from {filepath} trained for {checkpoint['epoch']} epochs.")
        return checkpoint

def train_gan(epoch_count: int, learning_rate: float, batch_size: int, train_existing: bool, test_only: bool, queue = None, queue_from_gui = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    
    # Files and folders.
    filepath_model = os.path.join(FOLDER_ROOT, "gan.pth")
    folder_results = os.path.join(FOLDER_ROOT, "Results")

    # Load the samples.
    samples = read_samples(os.path.join(FOLDER_ROOT, "samples.csv"))
    samples = samples.iloc[:1000, :]
    dataset = CantileverDataset(samples, is_3d=True)
    train_dataset = Subset(dataset, range(0, 800))
    validate_dataset = Subset(dataset, range(800, 1000))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    size_train_dataset = len(train_dataloader)
    size_validate_dataset = len(validate_dataloader)

    # Initialize the models.
    LATENT_SIZE = 100
    model_generator = GanGenerator(LATENT_SIZE, 64, OUTPUT_SIZE_3D[-1])
    model_discriminator = GanDiscriminator(64, OUTPUT_SIZE_3D[-1])
    model_generator.to(device)
    model_discriminator.to(device)

    # Initialize the optimizers and loss functions.
    # optimizer_cnn = torch.optim.SGD(params=model_cnn.parameters(), lr=learning_rate)
    optimizer_generator = torch.optim.Adam(params=model_generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(params=model_discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    loss_function_cnn = nn.MSELoss()
    loss_function_gan = nn.BCELoss()

    # Load their parameters if they have been saved previously.
    epoch = 1
    previous_training_loss_cnn = []
    previous_training_loss_generator = []
    previous_training_loss_discriminator = []
    previous_validation_loss_cnn = []
    previous_validation_loss_generator = []
    previous_validation_loss_discriminator = []
    if os.path.exists(filepath_model):
        if not test_only:
            if train_existing is None:
                train_existing = input(f"Continue training the model in {filepath_model}? [y/n] ") == "y"
        else:
            train_existing = True
        
        if train_existing:
            checkpoint = torch.load(filepath_model, map_location=torch.device(device))
            model_generator.load_state_dict(checkpoint["model_generator_state_dict"])
            model_discriminator.load_state_dict(checkpoint["model_discriminator_state_dict"])
            optimizer_generator.load_state_dict(checkpoint["optimizer_generator_state_dict"])
            optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator_state_dict"])
            epoch = checkpoint["epoch"] + 1
            previous_training_loss_generator = checkpoint["training_loss_generator"]
            previous_training_loss_discriminator = checkpoint["training_loss_discriminator"]
            previous_validation_loss_generator = checkpoint["validation_loss_generator"]
            previous_validation_loss_discriminator = checkpoint["validation_loss_discriminator"]
    else:
        train_existing = False
        test_only = False
    
    epochs = range(epoch, epoch+epoch_count)
    # if queue:
    #     queue.put([(epochs[0]-1, epochs[-1]), None, None, None, None])
    
    # Initialize the validation losses.
    training_loss_cnn = []
    training_loss_generator = []
    training_loss_discriminator = []
    validation_loss_cnn = []
    validation_loss_generator = []
    validation_loss_discriminator = []

    # Main training-validation loop.
    LABEL_REAL = 1
    LABEL_FAKE = 0
    for epoch in epochs:
        print(f"Epoch {epoch}\n------------------------")

        # model_cnn.train(True)
        model_generator.train(True)
        model_discriminator.train(True)

        # Train on the training dataset.
        loss_cnn = 0
        loss_generator = 0
        loss_discriminator = 0
        for batch, (_, image) in enumerate(train_dataloader, 1):
            image = image.to(device)
            image = image.float()

            label_real = torch.full((image.size(0),), LABEL_REAL, dtype=torch.float, device=device)
            label_fake = torch.full((image.size(0),), LABEL_FAKE, dtype=torch.float, device=device)

            # # Reset gradients.
            # optimizer_cnn.zero_grad()
            # optimizer_generator.zero_grad()
            # optimizer_discriminator.zero_grad()

            # Train the discriminator with an all-real batch.
            model_discriminator.zero_grad()
            # Forward pass real images through the discriminator.
            output_discriminator_real = model_discriminator(image).view(-1)
            # Calculate the loss of the discriminator.
            loss_real = loss_function_gan(output_discriminator_real, label_real)
            # Calculate gradients.
            loss_real.backward()

            # Train the discriminator with an all-fake batch.
            # output_cnn = model_cnn(data)
            output_cnn = torch.randn(image.size(0), LATENT_SIZE, 1, 1, device=device)
            output_generator = model_generator(output_cnn)
            # Forward pass fake images through the discriminator.
            output_discriminator_fake = model_discriminator(output_generator.detach()).view(-1)
            # Calculate the loss of the discriminator.
            loss_fake = loss_function_gan(output_discriminator_fake, label_fake)
            # Calculate gradients.
            loss_fake.backward()

            # Calculate total loss of discriminator by summing real and fake losses.
            loss_discriminator_current = loss_real + loss_fake
            loss_discriminator += loss_discriminator_current.item()
            # Update discriminator.
            optimizer_discriminator.step()

            # Train the generator.
            model_generator.zero_grad()
            # Forward pass fake images through the discriminator.
            output_discriminator = model_discriminator(output_generator).view(-1)
            # Calculate the loss of the generator, assuming images are real in order to calculate loss correctly.
            loss_generator_current = loss_function_gan(output_discriminator, label_real)
            loss_generator += loss_generator_current.item()
            # Calculate gradients.
            loss_generator_current.backward()
            # Update generator.
            optimizer_generator.step()

            # Train the CNN.
            # loss_cnn = loss_function_cnn(output_generator, label)
            # loss_cnn.backward()
            # optimizer_cnn.step()

            # Periodically display progress.
            if batch % 5 == 0:
                print(f"Training batch {batch}/{size_train_dataset}...", end="\r")
                if queue:
                    queue.put([None, (batch, size_train_dataset+size_validate_dataset), None, None, None])
        print()
        loss_cnn /= batch
        loss_generator /= batch
        loss_discriminator /= batch
        training_loss_cnn.append(loss_cnn)
        training_loss_generator.append(loss_generator)
        training_loss_discriminator.append(loss_discriminator)
        print(f"Average training loss: {loss_generator:,.2f} G, {loss_discriminator:,.2f} D, {loss_cnn:,.2f} CNN")
        
        # model_cnn.train(False)
        model_generator.train(False)
        model_discriminator.train(False)
        
        # Test on the validation dataset.
        loss_cnn = 0
        loss_generator = 0
        loss_discriminator = 0
        with torch.no_grad():
            for batch, (_, image) in enumerate(validate_dataloader, 1):
                image = image.to(device)
                image = image.float()

                output_cnn = torch.randn(image.size(0), LATENT_SIZE, 1, 1, device=device)  # model_cnn(data)
                output_generator = model_generator(output_cnn)

                # Write an output image periodically.
                if epoch % 10 == 0:
                    write_image(
                        np.hstack([output_cnn[0, channel, ...] for channel in output_cnn.shape[1]]),
                        f"gan_{batch}.png",
                    )
                
                output_discriminator_real = model_discriminator(image).view(-1)
                output_discriminator_fake = model_discriminator(output_generator).view(-1)

                label_real = torch.full((image.size(0),), LABEL_REAL, dtype=torch.float, device=device)
                label_fake = torch.full((image.size(0),), LABEL_FAKE, dtype=torch.float, device=device)

                # loss_cnn += loss_function_cnn(output_generator, ).item()
                loss_generator = loss_function_gan(output_discriminator_fake, label_real).item()
                loss_discriminator = loss_function_gan(output_discriminator_real, label_real).item() + loss_function_gan(output_discriminator_fake, label_fake).item()

                if batch % 5 == 0:
                    print(f"Validating batch {batch}/{size_validate_dataset}...", end="\r")
                    if queue:
                        queue.put([None, (size_train_dataset+batch, size_train_dataset+size_validate_dataset), None, None, None])
        print()
        loss_cnn /= batch
        loss_generator /= batch
        loss_discriminator /= batch
        validation_loss_cnn.append(loss_cnn)
        validation_loss_generator.append(loss_generator)
        validation_loss_discriminator.append(loss_discriminator)
        print(f"Average validation loss: {loss_generator:,.2f} G, {loss_discriminator:,.2f} D, {loss_cnn:,.2f} CNN")

        # Save the model parameters periodically.
        if epoch % 1 == 0:
            save_model(
                filepath_model,
                epoch=epoch,
                model_generator_state_dict=model_generator.state_dict(),
                model_discriminator_state_dict=model_discriminator.state_dict(),
                optimizer_generator_state_dict=optimizer_generator.state_dict(),
                optimizer_discriminator_state_dict=optimizer_discriminator.state_dict(),
                training_loss_generator=[*previous_training_loss_generator, *training_loss_generator],
                training_loss_discriminator=[*previous_training_loss_discriminator, *training_loss_discriminator],
                validation_loss_generator=[*previous_validation_loss_generator, *validation_loss_generator],
                validation_loss_discriminator=[*previous_validation_loss_discriminator, *validation_loss_discriminator],
            )
        
        # if queue:
        #     queue.put([(epoch, epochs[-1]), None, epochs, validation_losses, previous_validation_losses])
        
        # if queue_from_gui:
        #     if not queue_from_gui.empty():
        #         # Stop training.
        #         if queue_from_gui.get() == True:
        #             queue_from_gui.queue.clear()
        #             break
    
    # Plot the loss history.
    if not queue:
        plt.figure()
        if previous_validation_loss_cnn and previous_validation_loss_generator and previous_validation_loss_discriminator:
            # plt.plot(range(1, epochs[0]), previous_validation_losses_cnn, 'x', color=Colors.GRAY_LIGHT)
            plt.plot(range(1, epochs[0]), previous_validation_loss_generator, 'o', color=Colors.GRAY_LIGHT)
            plt.plot(range(1, epochs[0]), previous_validation_loss_discriminator, '*', color=Colors.GRAY_LIGHT)
        # plt.plot(epochs, validation_losses_cnn, '-x', color=Colors.BLUE, label="CNN")
        plt.plot(epochs, validation_loss_generator, '-o', color=Colors.BLUE, label="Generator")
        plt.plot(epochs, validation_loss_discriminator, '-*', color=Colors.ORANGE, label="Discriminator")
        plt.legend()
        plt.ylim(bottom=0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(axis="y")
        plt.show()

def train_cnn_encoder(epoch_count: int, learning_rate: float, batch_size: int, train_existing: bool, test_only: bool, queue = None, queue_from_gui = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    
    # Files and folders.
    filepath_model = os.path.join(FOLDER_ROOT, "cnn_encoder.pth")
    folder_results = os.path.join(FOLDER_ROOT, "Results")

def train_regression(
    device: str, epoch_count: int, checkpoint: dict, filepath_model: str,
    model: nn.Module, optimizer: torch.optim.Optimizer, loss_function: nn.Module,
    train_dataloader: DataLoader, validate_dataloader: DataLoader,
    queue=None, queue_to_main=None,
) -> nn.Module:
    """Train and validate the given regression model. Return the model after finishing training."""

    # Load the previous training history.
    if checkpoint is not None:
        epoch = checkpoint["epoch"] + 1
        previous_training_loss = checkpoint["training_loss"]
        previous_validation_loss = checkpoint["validation_loss"]
    else:
        epoch = 1
        previous_training_loss = []
        previous_validation_loss = []
    epochs = range(epoch, epoch+epoch_count)

    # Initialize values to send to the GUI, to be updated throughout training.
    if queue:
        info_gui = {
            "progress_epoch": (epoch, epochs[-1]),
            "progress_batch": (0, 0),
            "epochs": epochs,
            "training_loss": [],
            "previous_training_loss": previous_training_loss,
            "validation_loss": [],
            "previous_validation_loss": previous_validation_loss,
        }
        queue.put(info_gui)

    # Initialize the loss values for the current training session.
    training_loss = []
    validation_loss = []

    # Main training-validation loop.
    for epoch in epochs:
        print(f"\nEpoch {epoch} ({time.strftime('%I:%M:%S %p')})")
        
        # Train on the training dataset.
        model.train(True)
        loss = 0
        for batch, ((input_data, load), label_data) in enumerate(train_dataloader, 1):
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            
            # Predict an output from the model with the given input.
            output_data = model(input_data)
            # Calculate the loss.
            loss_current = loss_function(output_data, label_data.float())
            # Update the cumulative loss.
            loss += loss_current.item()
            
            # Reset gradients of model parameters.
            optimizer.zero_grad()
            # Backpropagate the prediction loss.
            loss_current.backward()
            # Adjust model parameters.
            optimizer.step()

            if batch % 50 == 0:
                print(f"Training batch {batch}/{len(train_dataloader)} with average loss {loss/batch:,.2f}...", end="\r")
                if queue:
                    info_gui["progress_batch"] = (batch, len(train_dataloader)+len(validate_dataloader))
                    info_gui["training_loss"] = [*training_loss, loss/batch]
                    queue.put(info_gui)
        print()
        loss /= batch
        training_loss.append(loss)
        print(f"Average training loss: {loss:,.2f}")

        # Test on the validation dataset. Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
        model.train(False)
        loss = 0
        with torch.no_grad():
            for batch, ((input_data, load), label_data) in enumerate(validate_dataloader, 1):
                input_data = input_data.to(device)
                label_data = label_data.to(device)
                output_data = model(input_data)
                loss += loss_function(output_data, label_data.float()).item()
                if batch % 50 == 0:
                    print(f"Validating batch {batch}/{len(validate_dataloader)}...", end="\r")
                    if queue:
                        info_gui["progress_batch"] = (len(train_dataloader)+batch, len(train_dataloader)+len(validate_dataloader))
                        queue.put(info_gui)
        print()
        loss /= batch
        validation_loss.append(loss)
        print(f"Average validation loss: {loss:,.2f}")

        # Save the model parameters periodically and in the last iteration of the loop.
        if epoch % 5 == 0 or epoch == epochs[-1]:
            save_model(
                filepath_model,
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                training_loss=[*previous_training_loss, *training_loss],
                validation_loss=[*previous_validation_loss, *validation_loss],
            )
        
        if queue:
            info_gui["progress_epoch"] = (epoch, epochs[-1])
            info_gui["training_loss"] = training_loss
            info_gui["validation_loss"] = validation_loss
            queue.put(info_gui)
        
        # Stop if the user stopped training from the GUI.
        if queue_to_main:
            if not queue_to_main.empty() and queue_to_main.get() == True:
                queue_to_main.queue.clear()
                break
    
    # Plot the loss history.
    if not queue:
        plt.figure()
        if previous_training_loss:
            plt.plot(range(1, epochs[0]), previous_training_loss, ".:", color=Colors.GRAY_LIGHT)
        if previous_validation_loss:
            plt.plot(range(1, epochs[0]), previous_validation_loss, ".-", color=Colors.GRAY_LIGHT)
        plt.plot(epochs, training_loss, ".:", color=Colors.ORANGE, label="Training")
        plt.plot(epochs, validation_loss, ".-", color=Colors.BLUE, label="Validation")
        plt.legend()
        plt.ylim(bottom=0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(axis="y")
        plt.show()
    
    return model

def test_regression(
    device: str, model: nn.Module, loss_function: nn.Module, test_dataloader: DataLoader,
    queue=None, queue_to_main=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test the given regression model and return its outputs and corresponding labels and inputs."""

    # Initialize values to send to the GUI.
    if queue:
        info_gui = {
            "progress_batch": (0, 0),
            "values_metrics": {},
        }
        queue.put(info_gui)

    model.train(False)
    
    loss = 0
    inputs = []
    outputs = []
    labels = []

    with torch.no_grad():
        for batch, ((input_data, load), label_data) in enumerate(test_dataloader, 1):
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            output_data = model(input_data)
            loss += loss_function(output_data, label_data.float()).item()

            # Convert to NumPy arrays for evaluation metric calculations.
            input_data = input_data.cpu().detach().numpy()
            output_data = output_data.cpu().detach().numpy()
            label_data = label_data.cpu().numpy()

            inputs.append(input_data)
            labels.append(label_data)
            outputs.append(output_data)
            
            if batch % 1 == 0:
                print(f"Testing batch {batch}/{len(test_dataloader)}...", end="\r")
                if queue:
                    info_gui["progress_batch"] = (batch, len(test_dataloader))
                    queue.put(info_gui)
        print()
    loss /= batch
    print(f"Average testing loss: {loss:,.2f}")
    
    # Concatenate testing results from all batches into a single array.
    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    labels = np.concatenate(labels, axis=0)

    return outputs, labels, inputs

def evaluate_regression(outputs: np.ndarray, labels: np.ndarray, inputs: np.ndarray, dataset: Dataset, queue=None):
    """Show and plot evaluation metrics."""
    folder_results = os.path.join(FOLDER_ROOT, "Results")

    me = metrics.mean_error(outputs, labels)
    # mae = metrics.mean_absolute_error(outputs, labels)
    # mse = metrics.mean_squared_error(outputs, labels)
    # rmse = metrics.root_mean_squared_error(outputs, labels)
    nmae = metrics.normalized_mean_absolute_error(outputs, labels)
    nmse = metrics.normalized_mean_squared_error(outputs, labels)
    nrmse = metrics.normalized_root_mean_squared_error(outputs, labels)
    mre = metrics.mean_relative_error(outputs, labels)
    print(f"ME: {me:,.2f}")
    print(f"NMAE: {nmae:,.2f}")
    print(f"NMSE: {nmse:,.2f}")
    print(f"NRMSE: {nrmse:,.2f}")
    print(f"MRE: {mre:,.2f}%")

    # Write output images and corresponding label images to files. Concatenate them vertically, and concatenate channels (if multiple channels) horizontally.
    indices = range(0, labels.shape[0], 100)
    for i in indices:
        image = np.vstack((
            np.hstack([labels[i, channel, ...] for channel in range(labels.shape[1])]),
            np.hstack([outputs[i, channel, ...] for channel in range(outputs.shape[1])]),
        ))
        write_image(
            array_to_colormap(image, dataset.max_value),
            os.path.join(folder_results, f"{i+1}_fea_model.png"),
            )
    print(f"Wrote {len(indices)} test images in {folder_results}.")

    # # Write images for each channel in the specified input images.
    # for i in (720, 1060, 2960):
    #     for channel in range(inputs.shape[1]):
    #         write_image(
    #             inputs[i, channel, ...],
    #             os.path.join(folder_results, f"input_{i+1}_channel_{channel+1}.png"),
    #         )

    # Plot 3D voxel models for corresponding model outputs and labels for the specified samples.
    if not queue and 1 not in labels.shape[-3:]:
        for i in (0,):
            TRANSPARENCY = 1.0

            fig = plt.figure()

            # Plot the predicted values.
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            # Array of RGBA values with shape (channel, height, length, 4).
            rgb = np.empty(outputs.shape[1:4] + (4,))
            if isinstance(dataset, CantileverDataset3d):
                rgb[..., :3] = array_to_colormap(outputs[i, 0, ...], dataset.max_value)
            else:
                rgb[..., :3] = array_to_colormap(outputs[i, ...], dataset.max_value)
            rgb /= 255
            # Set the transparency value.
            rgb[..., -1] = TRANSPARENCY
            # Boolean array representing the region of voxels to make visible.
            filled = labels[i, ...].transpose((2, 0, 1)) != 0
            # Plot the output's length on the X-axis (X in FEA), the width on the Y-axis (Z in FEA), and the height on the Z-axis (Y in FEA).
            ax.voxels(
                filled=filled,
                facecolors=rgb.transpose((2, 0, 1, 3)),
                linewidth=0.25,
                edgecolors=(1, 1, 1),
            )
            ax.set(xlabel="X (Length)", ylabel="Z (Width)", zlabel="Y (Height)")
            axis_limits = [0, max(outputs.shape[1:])]
            ax.set(xlim=axis_limits, ylim=axis_limits, zlim=axis_limits)
            ax.set_title("Predicted")
            
            # Plot the true values.
            ax = fig.add_subplot(1, 2, 2, projection="3d")
            rgb = np.empty(labels.shape[1:4] + (4,))
            if isinstance(dataset, CantileverDataset3d):
                rgb[..., :3] = array_to_colormap(labels[i, 0, ...], dataset.max_value)
            else:
                rgb[..., :3] = array_to_colormap(labels[i, ...], dataset.max_value)
            rgb /= 255
            rgb[..., -1] = TRANSPARENCY
            filled = labels[i, ...].transpose((2, 0, 1)) != 0
            ax.voxels(
                filled=filled,
                facecolors=rgb.transpose((2, 0, 1, 3)),
                linewidth=0.25,
                edgecolors=(1, 1, 1),
            )
            ax.set(xlabel="X (Length)", ylabel="Z (Width)", zlabel="Y (Height)")
            axis_limits = [0, max(labels.shape[1:])]
            ax.set(xlim=axis_limits, ylim=axis_limits, zlim=axis_limits)
            ax.set_title("True")
            
            plt.show()

    cdf_network, cdf_label, bin_edges, area_difference = metrics.area_metric(
        outputs.flatten(),
        labels.flatten(),
        dataset.max_value,
        plot=not queue,
    )

    max_network, max_label = metrics.maximum_value(outputs, labels, plot=not queue)

def main(
    train: bool, test: bool, evaluate: bool, train_existing: bool,
    epoch_count: int, learning_rate: float, batch_sizes: Tuple[int, int, int], Model: nn.Module,
    dataset_id: int, training_split: Tuple[float, float, float], filename_model: str, filename_subset: str,
    Optimizer: torch.optim.Optimizer = torch.optim.SGD, Loss: nn.Module = nn.MSELoss,
    queue: Queue = None, queue_to_main: Queue = None,
):
    """
    Function run directly by this file and by the GUI.

    Parameters:
    `train`: Train the model.
    `test`: Test the model.
    `evaluate`: Evaluate the test results, if testing.
    `train_existing`: Load a previously saved model and continue training it.

    `epoch_count`: Number of epochs to train.
    `learning_rate`: Learning rate for the optimizer.
    `batch_sizes`: Tuple of batch sizes for the training, validation, and testing datasets.
    `Model`: A Module subclass to instantiate, not an instance of the class.
    `Optimizer`: An Optimizer subclass to instantiate, not an instance of the class.
    `Loss`: A Module subclass to instantiate, not an instance of the class.

    `dataset_id`: An integer representing the dataset to use.
    `training_split`: A tuple of three floats in [0, 1] of the training, validation, and testing ratios.
    `filename_model`: Name of the .pth file to load and save to during training.
    `filename_subset`: Name of the .txt file that contains a subset of the entire dataset to use.
    
    `queue`: A Queue used to send information to the GUI.
    `queue_to_main`: A Queue used to receive information from the GUI.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    filepath_model = os.path.join(FOLDER_ROOT, filename_model)

    if (test and not train) or (train and train_existing):
        checkpoint = load_model(filepath=filepath_model, device=device)
    else:
        checkpoint = None

    assert dataset_id in {2, 3, 4}, f"Invalid dataset ID: {dataset_id}."
    
    # Load the samples.
    samples = read_samples(os.path.join(FOLDER_ROOT, "samples.csv"))
    samples = samples.iloc[:50000, :]

    # Get the specified subset of the dataset, if provided.
    if filename_subset is not None:
        filepath_subset = os.path.join(FOLDER_ROOT, filename_subset)
        with open(filepath_subset, "r") as f:
            sample_indices = [int(_) - 1 for _ in f.readlines()]
        
        samples = samples.iloc[sample_indices]
        print(f"Using a subset with {len(samples)} samples loaded from {filepath_subset}.")

    # Calculate the dataset split sizes.
    sample_size = len(samples)
    train_size, validate_size, test_size = [int(split * sample_size) for split in training_split]
    assert train_size + validate_size + test_size == sample_size
    print(f"Split {sample_size} samples into {train_size} training / {validate_size} validation / {test_size} test.")
    
    # Create the Dataset containing all data.
    if dataset_id == 2:
        dataset = CantileverDataset(samples, is_3d=False)
    elif dataset_id == 3:
        dataset = CantileverDataset(samples, is_3d=True)
    elif dataset_id == 4:
        dataset = CantileverDataset3d(samples)
    
    # Split the dataset into training, validation, and testing.
    batch_size_train, batch_size_validate, batch_size_test = batch_sizes
    train_dataset = Subset(dataset, range(0, train_size))
    validate_dataset = Subset(dataset, range(train_size, train_size+validate_size))
    test_dataset = Subset(dataset, range(train_size+validate_size, train_size+validate_size+test_size))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size_validate, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    # Initialize the model, optimizer, and loss function.
    args = {
        Nie: [dataset.input_channels, INPUT_SIZE, dataset.output_channels],
        Nie3d: [dataset.input_channels, INPUT_SIZE_3D, dataset.output_channels],
        FullyCnn: [dataset.input_channels, OUTPUT_SIZE_3D if dataset_id == 3 else OUTPUT_SIZE, dataset.output_channels],
        UNetCnn: [dataset.input_channels, dataset.output_channels],
        AutoencoderCnn: [dataset.input_channels, dataset.output_channels],
    }
    model = Model(*args[Model])
    model.to(device)
    optimizer = Optimizer(model.parameters(), lr=learning_rate)
    loss_function = Loss()

    # Load previously saved model and optimizer parameters.
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if train:
        model = train_regression(
            device = device,
            epoch_count = epoch_count,
            checkpoint = checkpoint,
            filepath_model = filepath_model,
            model = model,
            optimizer = optimizer,
            loss_function = loss_function,
            train_dataloader = train_dataloader,
            validate_dataloader = validate_dataloader,
            queue = queue,
            queue_to_main = queue_to_main,
            )
    
    if test:
        outputs, labels, inputs = test_regression(
            device = device,
            model = model,
            loss_function = loss_function,
            test_dataloader = test_dataloader,
            queue = queue,
            queue_to_main = queue_to_main,
        )

        # Transform values back to original range.
        outputs = dataset.transform(outputs, inverse=True)
        labels = dataset.transform(labels, inverse=True)

        if evaluate:
            evaluate_regression(outputs, labels, inputs, dataset, queue=queue)


if __name__ == "__main__":
    kwargs = {
        "epoch_count": 1,
        "learning_rate": 1e-3,
        "batch_sizes": (8, 128, 256),
        
        "Model": Nie,
        "dataset_id": 2,
        "training_split": (0.8, 0.1, 0.1),
        "filename_model": "model.pth",
        "filename_subset": None,

        "Optimizer": torch.optim.SGD,
        "Loss": nn.MSELoss,
        
        "train_existing": not True,
        "train": True,
        "test": True,
        "evaluate": True,
    }

    main(**kwargs)