"""
Train and test the model.
"""


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


# Model parameters file path.
FILEPATH_MODEL = os.path.join(FOLDER_ROOT, "model.pth")


class CantileverDataset2d(Dataset):
    """Dataset that contains input images and label images."""
    def __init__(self, samples: pd.DataFrame, folder_labels: str):
        self.number_samples = len(samples)
        
        # Load previously generated label images.
        files = glob.glob(os.path.join(folder_labels, "*.pickle"))
        if files:
            file = files[0]
            self.labels = read_pickle(file)
        # Create label images and save them as a pickle file.
        else:
            self.labels = generate_label_images(samples, folder_labels, is_3d=False)
            file = os.path.join(folder_labels, f"labels.pickle")
            write_pickle(self.labels, file)
        print(f"Label images take up {sys.getsizeof(self.labels)/1e9:,.2f} GB.")
        
        # Create input images.
        self.inputs = generate_input_images(samples, is_3d=False)
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
        if files:
            file = files[0]
            self.labels = read_pickle(file)
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

def main(epoch_count: int, learning_rate: float, batch_size: int, Model: nn.Module, dataset: int, training_split: Tuple[float, float, float], filename_subset: str = None, train_existing=None, test_only=False, queue=None, queue_to_main=None):
    """
    Train and test the model.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Files and folders.
    if dataset == 2:
        folder_labels = os.path.join(FOLDER_ROOT, "Labels")
    elif dataset == 3:
        folder_labels = os.path.join(FOLDER_ROOT, "Labels 3D")
    else:
        raise ValueError(f"Invalid dataset ID: {dataset}.")
    folder_results = os.path.join(FOLDER_ROOT, "Results")

    # Load the samples.
    samples = read_samples(os.path.join(FOLDER_ROOT, FILENAME_SAMPLES))
    samples = samples.iloc[:10000, :]

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
    if dataset == 2:
        dataset = CantileverDataset2d(samples, folder_labels)
    elif dataset == 3:
        dataset = CantileverDataset3d(samples, folder_labels)
    train_dataset = Subset(dataset, range(0, train_size))
    validation_dataset = Subset(dataset, range(train_size, train_size+validate_size))
    test_dataset = Subset(dataset, range(train_size+validate_size, train_size+validate_size+test_size))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    size_train_dataset = len(train_dataloader)
    size_validation_dataset = len(validation_dataloader)
    size_test_dataset = len(test_dataloader)

    # Initialize the model and optimizer and load their parameters if they have been saved previously.
    model_args = {
        Nie: [dataset.input_channels, INPUT_SIZE, dataset.output_channels],
        Nie3d: [dataset.input_channels, INPUT_SIZE_3D, dataset.output_channels],
        FullyCnn: [dataset.input_channels, OUTPUT_SIZE_3D if dataset == 3 else OUTPUT_SIZE, dataset.output_channels],
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
    if os.path.exists(FILEPATH_MODEL):
        if not test_only:
            if train_existing is None:
                train_existing = input(f"Continue training the model in {FILEPATH_MODEL}? [y/n] ") == "y"
        else:
            train_existing = True
        
        if train_existing:
            checkpoint = torch.load(FILEPATH_MODEL, map_location=torch.device(device))
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
            loss /= size_train_dataset
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
            loss /= size_validation_dataset
            validation_loss.append(loss)
            print(f"Average validation loss: {loss:,.2f}")

            # Save the model parameters periodically.
            if (epoch) % 1 == 0:
                save(
                    FILEPATH_MODEL,
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
            FILEPATH_MODEL,
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
    labels = []
    outputs = []
    with torch.no_grad():
        for batch, ((input_image, load), label_image) in enumerate(test_dataloader, 1):
            input_image = input_image.to(device)
            label_image = label_image.to(device)
            output = model(input_image)
            loss += loss_function(output, label_image.float()).item()
            
            # Concatenate the FEA and model output images and write it to a file.
            if dataset == 2:
                output = output[0, :, ...].cpu().detach().numpy()
                label_image = label_image[0, :, ...].cpu().numpy()
                image = np.vstack((
                    np.hstack([label_image[channel, ...] for channel in range(label_image.shape[0])]),  # FEA
                    np.hstack([output[channel, ...] for channel in range(output.shape[0])]),  # Model output
                ))
            elif dataset == 3:
                output = output[0, 0, ...].cpu().detach().numpy()
                label_image = label_image[0, 0, ...].cpu().numpy()
                image = np.vstack((
                    np.hstack([label_image[..., channel] for channel in range(label_image.shape[-1])]),  # FEA
                    np.hstack([output[..., channel] for channel in range(output.shape[-1])]),  # Model output
                ))
            write_image(
                array_to_colormap(image, max_value),
                os.path.join(folder_results, f"{batch}_fea_model.png"),
                )
            
            labels.append(label_image)
            outputs.append(output)
            
            # Plot a voxel model of the model output.
            if not queue and dataset == 3:
                if batch in {3, 14, 19}:
                    fig = plt.figure()
                    ax = fig.gca(projection="3d")
                    rgb = np.empty(output.shape + (3,))
                    for channel in range(output.shape[0]):
                        rgb[channel, :, :, :] = array_to_colormap(output[channel, ...], max_value)
                    rgb /= 255
                    voxels = ax.voxels(filled=np.full(output.transpose((2, 0, 1)).shape, True), facecolors=rgb.transpose((2, 0, 1, 3)))
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    # plt.zlabel("Z")
                    plt.show()
            
            if queue:
                info_gui["progress_epoch"] = (0, 0)
                info_gui["progress_batch"] = (batch, size_test_dataset)
                queue.put(info_gui)
    loss /= size_test_dataset
    print(f"Average testing loss: {loss:,.2f}")
    
    print(f"Wrote {size_test_dataset} test images in {folder_results}.")

    # Calculate and plot evaluation metrics.
    if not queue:
        # plt.rc('font', family='Source Code Pro', size=10.0, weight='semibold')

        # Area metric.
        cdf_network, cdf_label, bin_edges, area_difference = metrics.area_metric(
            np.array([_ for _ in outputs]).flatten(),
            np.array([_ for _ in labels]).flatten(),
            max_value
        )
        plt.figure()
        plt.plot(bin_edges[1:], cdf_network, "-", color=Colors.BLUE)
        plt.plot(bin_edges[1:], cdf_label, ":", color=Colors.RED)
        plt.legend(["CNN", "FEA"])
        plt.grid(visible=True, axis="y")
        plt.xticks([*plt.xticks()[0], max_value])
        # plt.yticks([0, 1])
        plt.title(f"{area_difference:0.2f}", fontsize=10, fontweight="bold")
        plt.show()

        # NUMBER_COLUMNS = 4
        # for i, (output, label_image) in enumerate(zip(outputs, labels)):
        #     plt.subplot(math.ceil(len(outputs) / NUMBER_COLUMNS), NUMBER_COLUMNS, i+1)
        #     cdf_network, cdf_label, bin_edges, area_difference = metrics.area_metric(output[channel, ...], label_image[channel, ...], max_value)
        #     plt.plot(bin_edges[1:], cdf_network, '-', color=Colors.BLUE)
        #     plt.plot(bin_edges[1:], cdf_label, ':', color=Colors.RED)
        #     if i == 0:
        #         plt.legend(["CNN", "FEA"])
        #     plt.grid(visible=True, axis='y')
        #     plt.xticks([0, max_value])
        #     plt.yticks([0, 1])
        #     plt.title(f"[#{i+1}] {area_difference:0.2f}", fontsize=10, fontweight='bold')
        # plt.suptitle(f"Area Metric", fontweight='bold')
        # plt.tight_layout()  # Increase spacing between subplots
        # plt.show()

        # Single-value error metrics.
        mv, me, mae, mse, mre = [], [], [], [], []
        results = {"Maximum Value": mv, "Mean Error": me, "Mean Absolute Error": mae, "Mean Squared Error": mse, "Mean Relative Error": mre}
        for output, label_image in zip(outputs, labels):
            mv.append(metrics.maximum_value(output, label_image))
            me.append(metrics.mean_error(output, label_image))
            mae.append(metrics.mean_absolute_error(output, label_image))
            mse.append(metrics.mean_squared_error(output, label_image))
            mre.append(metrics.mean_relative_error(output, label_image))
        
        sample_numbers = range(1, len(outputs)+1)
        plt.figure()
        for i, (metric, result) in enumerate(results.items()):
            plt.subplot(3, 2, i+1)
            plt.grid()
            if isinstance(result[0], tuple):
                plt.plot(sample_numbers, [_[1] for _ in result], 'o', color=Colors.RED, label="FEA")
                plt.plot(sample_numbers, [_[0] for _ in result], '.', color=Colors.BLUE, label="CNN")
            else:
                plt.plot(sample_numbers, result, '.', markeredgewidth=5, color=Colors.BLUE)
                average = np.mean(result)
                plt.axhline(average, color=Colors.BLUE_LIGHT, label=f"{average:.2f} average")
            plt.legend()
            plt.xlabel("Sample Number")
            plt.xticks([sample_numbers[0], sample_numbers[-1]])
            plt.title(metric)
        plt.show()


if __name__ == '__main__':
    kwargs={
        "epoch_count": 20,
        "learning_rate": 1e-7,
        "batch_size": 1,
        "Model": Nie3d,
        "dataset": 3,

        "desired_subset_size": 10_000,
        "bins": 1,
        "nonuniformity": 1.0,
        "training_split": (0.8, 0.1, 0.1),
        "train_existing": True,
        "test_only": False,
    }

    main(**kwargs)