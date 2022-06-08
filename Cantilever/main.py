"""
Train and test the model.
"""


import gc
import glob
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader

from datasets import *
import metrics
from networks import *
from setup import *


# Free memory between subsequent runs.
if GOOGLE_COLAB:
    gc.collect()


class CantileverDataset(Dataset):
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
        print(f"Label images take up {sys.getsizeof(self.labels)/1e9:,.2f} GB.")
        
        # Create input images.
        self.inputs = generate_input_images(samples, is_3d=is_3d)
        print(f"Input images take up {sys.getsizeof(self.inputs)/1e9:,.2f} GB.")

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

class CantileverDataset3d(Dataset):
    """Dataset that contains input images and label images."""
    def __init__(self, samples: pd.DataFrame, folder_labels: str):
        self.number_samples = len(samples)
        
        # Load previously generated label images.
        files = glob.glob(os.path.join(folder_labels, "*.pickle"))
        files.sort()
        if files:
            file = files[0]
            self.labels = read_pickle(file)
            # Transpose dimensions to form: (samples, 1, height (Y), length (X), width (Z))
            self.labels = np.expand_dims(self.labels, axis=1).transpose((0, 1, 3, 4, 2))
        # Create label images and save them as a pickle file.
        else:
            self.labels = generate_label_images_3d(samples, folder_labels)
            file = os.path.join(folder_labels, f"labels.pickle")
            write_pickle(self.labels, file)
        print(f"Label images take up {sys.getsizeof(self.labels)/1e9:,.2f} GB.")
        
        # Create input images.
        self.inputs = generate_input_images_3d(samples)
        print(f"Input images take up {sys.getsizeof(self.inputs)/1e9:,.2f} GB.")

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

def save(filepath: str, **kwargs) -> None:
    """Save model parameters to a file."""
    torch.save(kwargs, filepath)
    print(f"Saved model parameters to {filepath}.")

def main(epoch_count: int, learning_rate: float, batch_size: int, Model: nn.Module, dataset_id: int, training_split: Tuple[float, float, float], filename_subset: str = None, filename_model: str = None, train_existing=None, test_only=False, queue=None, queue_to_main=None):
    """
    Train and test the model.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')
    
    assert dataset_id in {2, 3}, f"Invalid dataset ID: {dataset_id}."

    # Files and folders.
    filepath_model = os.path.join(FOLDER_ROOT, filename_model)
    folder_results = os.path.join(FOLDER_ROOT, "Results")

    # Load the samples.
    samples = read_samples(os.path.join(FOLDER_ROOT, FILENAME_SAMPLES))
    samples = samples.iloc[:30000, :]

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
    
    # Create the training, validation, and testing dataloaders.
    dataset = CantileverDataset(samples, is_3d=dataset_id == 3)
    # if dataset_id == 2:
    #     dataset = CantileverDataset(samples, folder_labels)
    # elif dataset_id == 3:
    #     dataset = CantileverDataset3d(samples, folder_labels)
    train_dataset = Subset(dataset, range(0, train_size))
    validation_dataset = Subset(dataset, range(train_size, train_size+validate_size))
    test_dataset = Subset(dataset, range(train_size+validate_size, train_size+validate_size+test_size))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    size_train_dataset = len(train_dataloader)
    size_validation_dataset = len(validation_dataloader)
    size_test_dataset = len(test_dataloader)

    # Initialize the model and optimizer and load their parameters if they have been saved previously.
    model_args = {
        Nie: [dataset.input_channels, INPUT_SIZE, dataset.output_channels],
        Nie3d: [dataset.input_channels, INPUT_SIZE_3D, dataset.output_channels],
        FullyCnn: [dataset.input_channels, OUTPUT_SIZE_3D if dataset_id == 3 else OUTPUT_SIZE, dataset.output_channels],
        UNetCnn: [dataset.input_channels, dataset.output_channels],
        AutoencoderCnn: [dataset.input_channels, dataset.output_channels],
    }
    args = model_args[Model]
    model = Model(*args)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    epoch = 1
    previous_training_loss = []
    previous_validation_loss = []
    if os.path.exists(filepath_model):
        if not test_only:
            if train_existing is None:
                train_existing = input(f"Continue training the model in {filepath_model}? [y/n] ") == "y"
        else:
            train_existing = True
        
        if train_existing:
            checkpoint = torch.load(filepath_model, map_location=torch.device(device))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"] + 1
            previous_training_loss = checkpoint["training_loss"]
            previous_validation_loss = checkpoint["validation_loss"]
    else:
        train_existing = False
        test_only = False
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

    if not test_only:
        if queue:
            queue.put(info_gui)

        training_loss = []
        validation_loss = []
        for epoch in epochs:
            print(f"Epoch {epoch}\n------------------------")
            
            # Train on the training dataset.
            model.train(True)
            loss = 0
            for batch, ((input_image, load), label_image) in enumerate(train_dataloader, 1):
                input_image = input_image.to(device)
                label_image = label_image.to(device)
                output = model(input_image)
                
                loss_current = loss_function(output, label_image.float())
                loss += loss_current.item()
                # Reset gradients of model parameters.
                optimizer.zero_grad()
                # Backpropagate the prediction loss.
                loss_current.backward()
                # Adjust model parameters.
                optimizer.step()

                if (batch) % 100 == 0:
                    print(f"Training batch {batch}/{size_train_dataset} with average loss {loss/batch:,.2f}...", end="\r")
                    if queue:
                        info_gui["progress_batch"] = (batch, size_train_dataset+size_validation_dataset)
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
                for batch, ((input_image, load), label_image) in enumerate(validation_dataloader, 1):
                    input_image = input_image.to(device)
                    label_image = label_image.to(device)
                    output = model(input_image)
                    loss += loss_function(output, label_image.float()).item()
                    if (batch) % 100 == 0:
                        print(f"Validating batch {batch}/{size_validation_dataset}...", end="\r")
                        if queue:
                            info_gui["progress_batch"] = (size_train_dataset+batch, size_train_dataset+size_validation_dataset)
                            queue.put(info_gui)
            print()
            loss /= batch
            validation_loss.append(loss)
            print(f"Average validation loss: {loss:,.2f}")

            # Save the model parameters periodically.
            if (epoch) % 1 == 0:
                save(
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
            
            if queue_to_main:
                if not queue_to_main.empty():
                    # Stop training.
                    if queue_to_main.get() == True:
                        queue_to_main.queue.clear()
                        break
        
        # Save the model parameters.
        save(
            filepath_model,
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            training_loss=[*previous_training_loss, *training_loss],
            validation_loss=[*previous_validation_loss, *validation_loss],
        )
        
        # Plot the loss history.
        if not queue:
            plt.figure()
            if previous_training_loss:
                plt.plot(range(1, epochs[0]), previous_training_loss, '.:', color=Colors.GRAY_LIGHT)
            if previous_validation_loss:
                plt.plot(range(1, epochs[0]), previous_validation_loss, '.-', color=Colors.GRAY_LIGHT)
            plt.plot(epochs, training_loss, '.:', color=Colors.ORANGE)
            plt.plot(epochs, validation_loss, '.-', color=Colors.BLUE)
            plt.legend()
            plt.ylim(bottom=0)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(axis='y')
            plt.show()

    # The maximum value found in the entire dataset, used to normalize values for images.
    max_value = np.max(dataset.labels)

    # Test on the testing dataset.
    model.train(False)
    loss = 0
    outputs = []
    labels = []
    with torch.no_grad():
        for batch, ((input_image, load), label_image) in enumerate(test_dataloader, 1):
            input_image = input_image.to(device)
            label_image = label_image.to(device)
            output = model(input_image)
            loss += loss_function(output, label_image.float()).item()

            # Convert to NumPy arrays for evaluation metric calculations.
            output = output.cpu().detach().numpy()
            label_image = label_image.cpu().numpy()
            
            labels.append(label_image)
            outputs.append(output)
            
            if queue:
                info_gui["progress_epoch"] = (0, 0)
                info_gui["progress_batch"] = (batch, size_test_dataset)
                queue.put(info_gui)
    loss /= batch
    print(f"Average testing loss: {loss:,.2f}")
    
    # Concatenate testing results from all batches into a single array.
    outputs = np.concatenate(outputs, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Concatenate corresponding model output images and label images for specified samples and write them to files.
    indices = range(0, len(test_dataset), 10)
    for i in indices:
        image = np.vstack((
            np.hstack([labels[i, channel, ...] for channel in range(labels.shape[1])]),
            np.hstack([outputs[i, channel, ...] for channel in range(outputs.shape[1])]),
        ))
        # if dataset_id == 2:
        # elif dataset_id == 3:
        #     image = np.vstack((
        #         np.hstack([labels[i, 0, ..., channel] for channel in range(labels.shape[-1])]),
        #         np.hstack([outputs[i, 0, ..., channel] for channel in range(outputs.shape[-1])]),
        #     ))
        write_image(
            array_to_colormap(image, max_value),
            os.path.join(folder_results, f"{i+1}_fea_model.png"),
            )
    print(f"Wrote {len(indices)} test images in {folder_results}.")
    
    # Plot 3D voxel models of the specified model outputs.
    if not queue and dataset_id == 3:
        for i in {3, 14, 19}:
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            rgb = np.empty(outputs.shape[1:4] + (3,))  # Shape (channel, height, length, 3)
            for channel in range(outputs.shape[1]):
                rgb[channel, :, :, :] = array_to_colormap(outputs[i, channel, ...], max_value)
            rgb /= 255
            # Make an array with a True region with the height, length, and width of the current sample.
            filled = labels[i, ...].transpose((2, 0, 1)) != 0
            # Plot voxels using arrays of shape (X = length, Y = width, Z = height).
            voxels = ax.voxels(
                filled=filled,
                facecolors=rgb.transpose((2, 0, 1, 3)),
                linewidth=0.25,
                edgecolors=(1, 1, 1),
            )
            ax.set(xlabel="X", ylabel="Y", zlabel="Z")
            axis_limits = [0, max(outputs.shape[1:])]
            ax.set(xlim=axis_limits, ylim=axis_limits, zlim=axis_limits)
            plt.show()

    # Plot or print evaluation metrics.
    cdf_network, cdf_label, bin_edges, area_difference = metrics.area_metric(
        outputs.flatten(),
        labels.flatten(),
        max_value,
        plot=not queue,
    )

    max_network, max_label = metrics.maximum_value(outputs, labels, plot=not queue)

    mae = metrics.mean_absolute_error(output, label_image)
    mse = metrics.mean_squared_error(output, label_image)
    mre = metrics.mean_relative_error(output, label_image)
    print(f"Mean absolute error: {mae}")
    print(f"Mean squared error: {mse}")
    print(f"Mean relative error: {mre}%")

if __name__ == '__main__':
    kwargs={
        "epoch_count": 20,
        "learning_rate": 1e-7,
        "batch_size": 1,
        "Model": Nie,
        "dataset_id": 3,
        "training_split": (0.8, 0.1, 0.1),
        
        "filename_model": "model.pth",
        "train_existing": True,
        "test_only": False,
    }

    main(**kwargs)