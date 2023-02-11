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


def save_model(filepath: str, **kwargs) -> None:
    """Save model parameters to a file."""
    torch.save(kwargs, filepath)
    print(f"Saved model parameters to {filepath}.")

def load_model(filepath: str, device: str) -> dict:
    """Return a dictionary of model parameters from a file."""
    try:
        checkpoint = torch.load(filepath, map_location=device)
    except FileNotFoundError:
        print(f"{filepath} not found.")
    else:
        print(f"Loaded model from {filepath} trained for {checkpoint['epoch']} epochs.")
        return checkpoint

def train_regression(
    device: str, epoch_count: int, checkpoint: dict, filepath_model: str, save_model_every: int,
    model: nn.Module, optimizer: torch.optim.Optimizer, loss_function: nn.Module,
    train_dataloader: DataLoader, validate_dataloader: DataLoader,
    scheduler = None,
    queue=None, queue_to_main=None, info_gui: dict=None,
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
        info_gui["progress_epoch"] = (epoch, epochs[-1])
        info_gui["progress_batch"] = (0, 0)
        info_gui["epochs"] = epochs
        info_gui["training_loss"] = []
        info_gui["previous_training_loss"] = previous_training_loss
        info_gui["validation_loss"] = []
        info_gui["previous_validation_loss"] = previous_validation_loss
        queue.put(info_gui)

    # Initialize the loss values for the current training session.
    training_loss = []
    validation_loss = []

    # Main training-validation loop.
    for epoch in epochs:
        print(f"\nEpoch {epoch}/{epochs[-1]} ({time.strftime('%I:%M %p')})")
        time_start = time.time()
        
        # Train on the training dataset.
        model.train(True)
        loss = 0
        for batch, (input_data, label_data) in enumerate(train_dataloader, 1):
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            
            # Predict an output from the model with the given input.
            output_data = model(input_data)
            # Calculate the loss.
            loss_current = loss_function(output_data, label_data)
            # Update the cumulative loss.
            loss += loss_current.item()

            if loss_current is torch.nan:
                print(f"Stopping due to nan loss.")
                break
            
            # Reset gradients of model parameters.
            optimizer.zero_grad()
            # Backpropagate the prediction loss.
            loss_current.backward()
            # Adjust model parameters.
            optimizer.step()

            if batch % 10 == 0:
                print(f"Batch {batch}/{len(train_dataloader)}: {loss/batch:,.2e}...", end="\r")
                if queue:
                    info_gui["progress_batch"] = (batch, len(train_dataloader)+len(validate_dataloader))
                    info_gui["training_loss"] = [*training_loss, loss/batch]
                    queue.put(info_gui)
            
            # Requested to stop from GUI.
            if queue_to_main and not queue_to_main.empty():
                break
        print()
        loss /= batch
        training_loss.append(loss)
        print(f"Training loss: {loss:,.2e}")

        # Adjust the learning rate if a scheduler is used.
        if scheduler:
            scheduler.step()
            learning_rate = optimizer.param_groups[0]["lr"]
            print(f"Learning rate: {learning_rate}")
            if queue:
                info_gui["info_training"]["Learning Rate"] = learning_rate
                queue.put(info_gui)

        # Test on the validation dataset. Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
        model.train(False)
        loss = 0
        outputs = []
        labels = []
        with torch.no_grad():
            for batch, (input_data, label_data) in enumerate(validate_dataloader, 1):
                input_data = input_data.to(device)
                label_data = label_data.to(device)
                output_data = model(input_data)
                loss += loss_function(output_data, label_data.float()).item()

                # Convert to NumPy arrays for evaluation metric calculations.
                output_data = output_data.cpu()
                label_data = label_data.cpu()

                outputs.append(output_data)
                labels.append(label_data)

                if batch % 50 == 0:
                    print(f"Batch {batch}/{len(validate_dataloader)}...", end="\r")
                    if queue:
                        info_gui["progress_batch"] = (len(train_dataloader)+batch, len(train_dataloader)+len(validate_dataloader))
                        queue.put(info_gui)
                
                # Requested to stop from GUI.
                if queue_to_main and not queue_to_main.empty():
                    break
        print()
        loss /= batch
        validation_loss.append(loss)
        print(f"Validation loss: {loss:,.2e}")

        # Calculate evaluation metrics on validation results.
        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        mre = metrics.mean_relative_error(outputs, labels)
        print(f"MRE: {mre:.3f}")

        # Save the model parameters periodically and in the last iteration of the loop.
        if epoch % save_model_every == 0 or epoch == epochs[-1]:
            save_model(
                filepath_model,
                epoch = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state_dict = optimizer.state_dict(),
                learning_rate = optimizer.param_groups[0]["lr"],
                training_loss = [*previous_training_loss, *training_loss],
                validation_loss = [*previous_validation_loss, *validation_loss],
            )
        
        # Show the elapsed time during the epoch.
        time_end = time.time()
        duration = time_end - time_start
        if duration >= 60:
            duration_text = f"{duration/60:.1f} minutes"
        else:
            duration_text = f"{duration:.1f} seconds"
        print(f"Finished epoch {epoch} in {duration_text}.")

        if queue:
            info_gui["progress_epoch"] = (epoch, epochs[-1])
            info_gui["training_loss"] = training_loss
            info_gui["validation_loss"] = validation_loss
            info_gui["info_training"]["Epoch Runtime"] = duration_text
            queue.put(info_gui)
        
        # Requested to stop from GUI.
        if queue_to_main and not queue_to_main.empty():
            queue_to_main.queue.clear()
            break
    
    # Plot the loss history.
    if not queue:
        figure = plt.figure()
        
        all_training_loss = [*previous_training_loss, *training_loss]
        all_validation_loss = [*previous_validation_loss, *validation_loss]
        plot_loss(figure, range(1, epochs[-1]+1), [all_training_loss, all_validation_loss], ["Training", "Validation"], start_epoch=epochs[0])

        plt.show()
    
    return model

def test_regression(
    device: str, model: nn.Module, loss_function: nn.Module, dataset: Dataset, test_dataloader: DataLoader,
    queue=None, queue_to_main=None, info_gui: dict=None,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Test the given regression model and return its outputs and corresponding labels and inputs."""

    # Initialize values to send to the GUI.
    if queue:
        info_gui["progress_batch"] = (0, 0)
        info_gui["info_metrics"] = {}
        queue.put(info_gui)

    model.train(False)
    
    loss = 0
    inputs = []
    outputs = []
    labels = []

    with torch.no_grad():
        for batch, (input_data, label_data) in enumerate(test_dataloader, 1):
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            output_data = model(input_data)
            loss += loss_function(output_data, label_data.float()).item()

            # Convert to NumPy arrays for evaluation metric calculations.
            input_data = input_data.cpu().detach()
            output_data = output_data.cpu().detach()
            label_data = label_data.cpu()

            inputs.append(input_data)
            labels.append(label_data)
            outputs.append(output_data)
            
            if batch % 1 == 0:
                print(f"Batch {batch}/{len(test_dataloader)}...", end="\r")
                if queue:
                    info_gui["progress_batch"] = (batch, len(test_dataloader))
                    queue.put(info_gui)
        print()
    loss /= batch
    print(f"Testing loss: {loss:,.2e}")

    # Show an output and corresponding label.
    i = 2
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(label_data[i, 0, ...])
    plt.title('True')
    plt.subplot(1, 2, 2)
    plt.imshow(output_data[i, 0, ...])
    plt.title('Predicted')
    plt.show()
    
    # Concatenate testing results from all batches into a single array.
    inputs = torch.cat(inputs, dim=0)
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    if queue:
        info_gui["info_metrics"] = {f"Loss ({loss_function})": loss}
        info_gui["test_inputs"] = inputs
        info_gui["test_outputs"] = outputs
        info_gui["test_labels"] = labels
        info_gui["test_max_value"] = dataset.max_value
        queue.put(info_gui)

    return outputs, labels, inputs

def evaluate_regression(outputs: np.ndarray, labels: np.ndarray, inputs: np.ndarray, dataset: Dataset, queue=None, info_gui: dict=None):
    """Calculate and return evaluation metrics."""
    folder_results = os.path.join(FOLDER_ROOT, "Results")

    results = {
        "ME": metrics.mean_error(outputs, labels),
        "MAE": metrics.mean_absolute_error(outputs, labels),
        "MSE": metrics.mean_squared_error(outputs, labels),
        "RMSE": metrics.root_mean_squared_error(outputs, labels),
        "NMAE": metrics.normalized_mean_absolute_error(outputs, labels),
        "NMSE": metrics.normalized_mean_squared_error(outputs, labels),
        "NRMSE": metrics.normalized_root_mean_squared_error(outputs, labels),
        "MRE": metrics.mean_relative_error(outputs, labels),
    }
    for metric, value in results.items():
        print(f"{metric}: {value:,.3f}")

    # # Write output images and corresponding label images to files. Concatenate them vertically, and concatenate channels (if multiple channels) horizontally.
    # indices = range(0, labels.shape[0], 100)
    # for i in indices:
    #     image = np.vstack((
    #         np.hstack([labels[i, channel, ...] for channel in range(labels.shape[1])]),
    #         np.hstack([outputs[i, channel, ...] for channel in range(outputs.shape[1])]),
    #     ))
    #     write_image(
    #         array_to_colormap(image, dataset.max_value),
    #         os.path.join(folder_results, f"{i+1}_fea_model.png"),
    #         )
    # print(f"Wrote {len(indices)} test images in {folder_results}.")

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

    # cdf_network, cdf_label, bin_edges, area_difference = metrics.area_metric(
    #     outputs.flatten(),
    #     labels.flatten(),
    #     dataset.max_value,
    #     plot=not queue,
    # )

    # max_network, max_label = metrics.maximum_value(outputs, labels, plot=not queue)

    # Initialize values to send to the GUI.
    if queue:
        info_gui["info_metrics"] = results
        queue.put(info_gui)
    
    return results

def main(
    epoch_count: int, learning_rate: float, decay_learning_rate: bool, batch_sizes: Tuple[int, int, int], dataset_split: Tuple[float, float, float], Model: nn.Module,
    filename_model: str, train_existing: bool, save_model_every: int,
    dataset_id: int,
    train: bool, test: bool, evaluate: bool,
    Optimizer: torch.optim.Optimizer = torch.optim.SGD, Loss: nn.Module = nn.MSELoss,
    normalize_inputs: bool = False, transformation_exponent: float = None,
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
    `dataset_split`: A tuple of three floats in [0, 1] of the training, validation, and testing ratios.
    `filename_model`: Name of the .pth file to load and save to during training.
    
    `queue`: A Queue used to send information to the GUI.
    `queue_to_main`: A Queue used to receive information from the GUI.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    # Initialize values to send to the GUI.
    info_gui = {
        "info_training": {},
        "info_metrics": {},
    } if queue else None

    filepath_model = os.path.join(FOLDER_CHECKPOINTS, filename_model)

    if (test and not train) or (train and train_existing):
        checkpoint = load_model(filepath=filepath_model, device=device)
        # Load the last learning rate used.
        if checkpoint and decay_learning_rate:
            learning_rate = checkpoint["learning_rate"]
            print(f"Using last learning rate {learning_rate}.")
    else:
        checkpoint = None

    assert dataset_id in {2, 3, 4}, f"Invalid dataset ID: {dataset_id}."

    # Load the samples.
    samples = read_samples(os.path.join(FOLDER_ROOT, "samples.csv"))

    # Calculate the dataset split sizes.
    train_size, validate_size, test_size = split_dataset(len(samples), dataset_split)
    print(f"Split {len(samples):,} samples into {train_size:,} training / {validate_size:,} validation / {test_size:,} test.")

    # Create the Dataset containing all data.
    if dataset_id == 2:
        if transformation_exponent is None:
            transformation_exponent = 0.4949464243559395
        dataset = CantileverDataset(samples, is_3d=False, normalize_inputs=normalize_inputs, transformation_exponent=transformation_exponent)
    elif dataset_id == 3:
        if transformation_exponent is None:
            transformation_exponent = 0.5023404737562848
        dataset = CantileverDataset(samples, is_3d=True, normalize_inputs=normalize_inputs, transformation_exponent=transformation_exponent)
    elif dataset_id == 4:
        if transformation_exponent is None:
            transformation_exponent = 0.5023404737562848
        dataset = CantileverDataset3d(samples, normalize_inputs=normalize_inputs, transformation_exponent=transformation_exponent)

    indices_train = range(0, train_size)
    indices_validate = range(train_size, train_size+validate_size)
    indices_test = range(train_size+validate_size, train_size+validate_size+test_size)

    results = []

    # Split the dataset into training, validation, and testing.
    batch_size_train, batch_size_validate, batch_size_test = batch_sizes
    train_dataset = Subset(dataset, indices_train)
    validate_dataset = Subset(dataset, indices_validate)
    test_dataset = Subset(dataset, indices_test)
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
    if decay_learning_rate:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        scheduler = None
    loss_function = Loss()

    # Load previously saved model and optimizer parameters.
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if queue:
            queue.put({
                "epochs": range(1, checkpoint["epoch"]+1),
                "training_loss": checkpoint["training_loss"],
                "validation_loss": checkpoint["validation_loss"],
            })

    if queue:
        info_gui["info_training"]["Training Size"] = train_size
        info_gui["info_training"]["Validation Size"] = validate_size
        info_gui["info_training"]["Testing Size"] = test_size
        info_gui["info_training"]["Learning Rate"] = learning_rate
        queue.put(info_gui)

    if train:
        model = train_regression(
            device = device,
            epoch_count = epoch_count,
            checkpoint = checkpoint,
            filepath_model = filepath_model,
            save_model_every = save_model_every,
            model = model,
            optimizer = optimizer,
            loss_function = loss_function,
            train_dataloader = train_dataloader,
            validate_dataloader = validate_dataloader,
            scheduler = scheduler,
            queue = queue,
            queue_to_main = queue_to_main,
            info_gui = info_gui,
            )

    if test:
        outputs, labels, inputs = test_regression(
            device = device,
            model = model,
            loss_function = loss_function,
            dataset = dataset,
            test_dataloader = test_dataloader,
            queue = queue,
            queue_to_main = queue_to_main,
            info_gui = info_gui,
        )

        # Transform values back to original range.
        dataset.scale(outputs, inverse=True)
        dataset.transform(outputs, inverse=True)
        dataset.scale(labels, inverse=True)
        dataset.transform(labels, inverse=True)

        # # Save a 3x6x3 stress prediction. Index bounds below are hardcoded from one specific output.
        # outputs = outputs[:, np.linspace(0, 9, 3).round(), ...]
        # outputs = outputs[:, :, np.linspace(0, 9, 3).round(), ...]  # 9 not 12; don't include the random small values
        # outputs = outputs[:, :, :, np.linspace(0, 20, 6).round()]
        # write_pickle(outputs[0, ...].numpy().transpose((1, 2, 0)), 'stress.pickle')

        outputs = outputs.numpy()
        labels = labels.numpy()
        inputs = inputs.numpy()

        if evaluate:
            results.append(
                evaluate_regression(outputs, labels, inputs, dataset, queue=queue, info_gui=info_gui)
            )

    return results


if __name__ == "__main__":
    kwargs = {
        "filename_model": "model.pth",
        "train_existing": not True,
        "save_model_every": 1,

        "epoch_count": 10,
        "learning_rate": 1e-5,
        "decay_learning_rate": not True,
        "batch_sizes": (16, 128, 128),
        "dataset_split": (0.8, 0.1, 0.1),
        "Model": Nie,
        "Optimizer": torch.optim.SGD,
        "Loss": nn.MSELoss,
        
        "dataset_id": 2,
        "normalize_inputs": False,
        "transformation_exponent": 1,
        
        "train": True,
        "test": True,
        "evaluate": True,
    }

    results = main(**kwargs)
