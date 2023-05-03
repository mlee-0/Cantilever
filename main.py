"""Run this script to train and test the model."""


import gc
import os
from queue import Queue
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import *
from preprocessing import *
import metrics
from networks import *


try:
    from google.colab import drive  # type: ignore (forces Pylance in VS Code to ignore the missing import error)
except ModuleNotFoundError:
    GOOGLE_COLAB = False
else:
    GOOGLE_COLAB = True
    drive.mount("/content/drive")

FOLDER_ROOT = "." if not GOOGLE_COLAB else "drive/My Drive/Colab Notebooks"
FOLDER_CHECKPOINTS = os.path.join(FOLDER_ROOT, "Checkpoints")

# Free memory between subsequent runs.
if GOOGLE_COLAB:
    gc.collect()


def save_model(filepath: str, **kwargs) -> None:
    """Save model parameters to a file."""
    torch.save(kwargs, filepath)
    print(f"Saved model parameters to {filepath}.")

def load_model(filepath: str, device: str='cpu') -> dict:
    """Return a dictionary of model parameters from a file."""
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Loaded model from {filepath} trained for {checkpoint['epoch']} epochs.")
    return checkpoint

def split_dataset(dataset_size: int, splits: List[float]) -> List[int]:
    """Return the subset sizes according to the fractions defined in `splits`."""

    assert sum(splits) == 1.0, f"The fractions {splits} must sum to 1."

    # Define the last subset size as the remaining number of data to ensure that they all sum to dataset_size.
    subset_sizes = []
    for fraction in splits[:-1]:
        subset_sizes.append(int(fraction * dataset_size))
    subset_sizes.append(dataset_size - sum(subset_sizes))

    return subset_sizes

def plot_loss(losses_training: List[float], losses_validation: List[float]) -> None:
    plt.figure()
    plt.semilogy(range(1, len(losses_training)+1), losses_training, '-', label='Training')
    plt.semilogy(range(1, len(losses_validation)+1), losses_validation, '-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_model(
    device: str, epoch_count: int, checkpoint: dict, filepath_model: str, save_model_every: int, save_best_separately: bool,
    model: nn.Module, optimizer: torch.optim.Optimizer, loss_function: nn.Module,
    train_dataloader: DataLoader, validate_dataloader: DataLoader,
    scheduler = None,
    queue=None, queue_to_main=None, info_gui: dict=None,
) -> nn.Module:
    """Train and validate the given regression model. Return the model after finishing training."""

    # Load the previous training history.
    epoch = checkpoint.get('epoch', 0) + 1
    epochs = range(epoch, epoch+epoch_count)

    training_loss = checkpoint.get('training_loss', [])
    validation_loss = checkpoint.get('validation_loss', [])

    # Initialize values to send to the GUI, to be updated throughout training.
    if queue:
        info_gui["progress_epoch"] = (epoch, epochs[-1])
        info_gui["progress_batch"] = (0, 0)
        info_gui["epochs"] = epochs
        info_gui["training_loss"] = training_loss
        info_gui["validation_loss"] = validation_loss
        queue.put(info_gui)

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
            # Calculate gradients.
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

                output_data = output_data.cpu()
                label_data = label_data.cpu()

                outputs.append(output_data)
                labels.append(label_data)

                if batch % 10 == 0:
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

        # # Calculate evaluation metrics on validation results.
        # outputs = torch.cat(outputs, dim=0)
        # labels = torch.cat(labels, dim=0)
        # outputs = dataset.untransform(outputs)
        # labels = dataset.untransform(labels)
        # evaluate_results(outputs.numpy(), labels.numpy())

        # Save the model periodically and in the last epoch.
        if epoch % save_model_every == 0 or epoch == epochs[-1]:
            save_model(
                filepath_model,
                epoch = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state_dict = optimizer.state_dict(),
                learning_rate = optimizer.param_groups[0]["lr"],
                training_loss = training_loss,
                validation_loss = validation_loss,
            )
        # Save the model if the model achieved the lowest validation loss so far.
        if save_best_separately and validation_loss[-1] <= min(validation_loss):
            save_model(
                f"{filepath_model[:-4]}[best]{filepath_model[-4:]}",
                epoch = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state_dict = optimizer.state_dict(),
                learning_rate = optimizer.param_groups[0]['lr'],
                training_loss = training_loss,
                validation_loss = validation_loss,
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

    return model

def test_model(
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

def evaluate_results(outputs: np.ndarray, labels: np.ndarray, queue=None, info_gui: dict=None):
    """Calculate and return evaluation metrics."""

    outputs_maxima = np.max(outputs, axis=tuple(range(1, outputs.ndim)))
    labels_maxima = np.max(labels, axis=tuple(range(1, labels.ndim)))

    results = {
        "ME": metrics.mean_error(outputs, labels),
        "MAE": metrics.mean_absolute_error(outputs, labels),
        "MSE": metrics.mean_squared_error(outputs, labels),
        "RMSE": metrics.root_mean_squared_error(outputs, labels),
        "NMAE": metrics.normalized_mean_absolute_error(outputs, labels),
        "NMSE": metrics.normalized_mean_squared_error(outputs, labels),
        "NRMSE": metrics.normalized_root_mean_squared_error(outputs, labels),
        "MRE": metrics.mean_relative_error(outputs, labels),

        "Maxima ME": metrics.mean_error(outputs_maxima, labels_maxima),
        "Maxima MAE": metrics.mean_absolute_error(outputs_maxima, labels_maxima),
        "Maxima MSE": metrics.mean_squared_error(outputs_maxima, labels_maxima),
        "Maxima RMSE": metrics.root_mean_squared_error(outputs_maxima, labels_maxima),
        "Maxima NMAE": metrics.normalized_mean_absolute_error(outputs_maxima, labels_maxima),
        "Maxima NMSE": metrics.normalized_mean_squared_error(outputs_maxima, labels_maxima),
        "Maxima NRMSE": metrics.normalized_root_mean_squared_error(outputs_maxima, labels_maxima),
        "Maxima MRE": metrics.mean_relative_error(outputs_maxima, labels_maxima),
    }
    for metric, value in results.items():
        print(f"{metric}: {value:,.3f}")

    # max_network, max_label = metrics.maximum_value(outputs, labels, plot=not queue)

    # Initialize values to send to the GUI.
    if queue:
        info_gui["info_metrics"] = results
        queue.put(info_gui)

    return results

def save_predictions(outputs: np.ndarray, labels: np.ndarray, inputs: np.ndarray, dataset: Dataset, save_every: int=1) -> None:
    """Save output data and corresponding label data as image files. They are concatenated vertically, and their channels (if multiple channels) are concatenated horizontally."""

    folder_results = os.path.join(FOLDER_ROOT, "Results")

    indices = range(0, labels.shape[0], save_every)
    for i in indices:
        image = np.vstack((
            np.hstack([labels[i, channel, ...] for channel in range(labels.shape[1])]),
            np.hstack([outputs[i, channel, ...] for channel in range(outputs.shape[1])]),
        ))
        write_image(
            array_to_colormap(image, dataset.max_value),
            os.path.join(folder_results, f"{i+1}_fea_model.png"),
            )

    print(f"Saved {len(indices)} test images in {folder_results}.")

def show_predictions_3d(outputs: np.ndarray, labels: np.ndarray, inputs: np.ndarray, dataset: Dataset, queue: Queue) -> None:
    """Plot 3D voxel models for output data and cooresponding label data."""

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

def main(
    epoch_count: int, learning_rate: float, decay_learning_rate: bool, batch_sizes: Tuple[int, int, int], dataset_split: Tuple[float, float, float], model: nn.Module, dataset: Dataset,
    filename_model: str, train_existing: bool, save_model_every: int, save_best_separately: bool,
    train: bool, test: bool, show_loss: bool, show_parity: bool, show_predictions: bool,
    Optimizer: torch.optim.Optimizer = torch.optim.SGD, Loss: nn.Module = nn.MSELoss,
    queue: Queue = None, queue_to_main: Queue = None,
):
    """
    Train and test a model.

    Inputs:
    `epoch_count`: Number of epochs to train.
    `learning_rate`: Learning rate.
    `decay_learning_rate`: Use a learning rate scheduler.
    `batch_sizes`: Tuple of batch sizes for the training, validation, and testing datasets.
    `dataset_split`: A tuple of three ratios in [0, 1] for the training, validation, and testing datasets.
    `model`: The network to train.
    `Optimizer`: An Optimizer subclass to instantiate, not an instance of the class.
    `Loss`: A Module subclass to instantiate, not an instance of the class.
    `dataset`: The dataset to train on.

    `filename_model`: Name of the .pth file to load and save to during training.
    `train_existing`: Load a previously saved model.
    `save_model_every`: Number of epochs between each instance of saving the model.
    `save_best_separately`: Track the best model (as evaluated on the validation dataset) and save it as a separate file.

    `train`: Train the model. Set to False to test a pretrained model.
    `test`: Test the model.
    `show_loss`: Plot the loss history.
    `show_parity`: Plot model predictions vs. labels.
    `show_predictions`: Show randomly selected model predictions with corresponding labels.
    """

    device = 'cpu'  #"cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {device} device.")

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
            learning_rate = checkpoint.get('learning_rate', learning_rate)
    else:
        checkpoint = {}

    # Split the dataset into training, validation, and testing.
    train_dataset, validate_dataset, test_dataset = random_split(
        dataset,
        split_dataset(len(dataset), dataset_split),
        generator=torch.Generator().manual_seed(42),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes[0], shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_sizes[1], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sizes[2], shuffle=False)
    print(f"Split {len(dataset):,} data into {len(train_dataset):,} training / {len(validate_dataset):,} validation / {len(test_dataset):,} test.")

    results = None

    # Initialize the model, optimizer, and loss function.
    model.to(device)
    optimizer = Optimizer(model.parameters(), lr=learning_rate)
    if decay_learning_rate:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        scheduler = None
    loss_function = Loss()

    # Load previously saved model and optimizer parameters.
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if queue:
            queue.put({
                "epochs": range(1, checkpoint["epoch"]+1),
                "training_loss": checkpoint["training_loss"],
                "validation_loss": checkpoint["validation_loss"],
            })

    if queue:
        info_gui["info_training"]["Training Size"] = len(train_dataset)
        info_gui["info_training"]["Validation Size"] = len(validate_dataset)
        info_gui["info_training"]["Testing Size"] = len(test_dataset)
        info_gui["info_training"]["Learning Rate"] = learning_rate
        queue.put(info_gui)

    if train:
        model = train_model(
            device = device,
            epoch_count = epoch_count,
            checkpoint = checkpoint,
            filepath_model = filepath_model,
            save_model_every = save_model_every,
            save_best_separately = save_best_separately,
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

    # Show the loss history.
    if show_loss:
        checkpoint = load_model(filepath=filepath_model)
        losses_training = checkpoint.get('training_loss', [])
        losses_validation = checkpoint.get('validation_loss', [])
        plot_loss(losses_training, losses_validation)

    # Load the best model.
    checkpoint = load_model(f"{filepath_model[:-4]}[best]{filepath_model[-4:]}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if test:
        outputs, labels, inputs = test_model(
            device = device,
            model = model,
            loss_function = loss_function,
            dataset = dataset,
            test_dataloader = test_dataloader,
            queue = queue,
            queue_to_main = queue_to_main,
            info_gui = info_gui,
        )

        # Transform values back to the original range.
        outputs = dataset.untransform(outputs)
        labels = dataset.untransform(labels)

        # Calculate evaluation metrics.
        results = evaluate_results(outputs.numpy(), labels.numpy(), queue=queue, info_gui=info_gui)

        # Show a parity plot.
        if show_parity:
            plt.plot(labels.flatten(), outputs.flatten(), '.')
            plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--')
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.show()

        # Show corresponding inputs, outputs, labels.
        if show_predictions:
            for i in random.sample(range(len(test_dataset)), k=3):
                label, output = labels[i, 0, ...], outputs[i, 0, ...]
                maximum = torch.max(label.max(), output.max())

                plt.figure(figsize=(6, 1.5))

                # plt.subplot(1, 4, 1)
                # plt.imshow(inputs[i, 0, ...], cmap='gray')
                # plt.subplot(1, 4, 2)
                # plt.imshow(inputs[i, 1, ...], cmap='gray')

                plt.subplot(1, 2, 1)
                plt.imshow(output, cmap='Spectral_r', vmin=0, vmax=maximum)
                plt.xticks([])
                plt.yticks([])
                plt.title('Predicted')
                colorbar = plt.colorbar(ticks=[0, maximum], fraction=0.05, aspect=10)
                colorbar.ax.tick_params(labelsize=6)

                plt.subplot(1, 2, 2)
                plt.imshow(label, cmap='Spectral_r', vmin=0, vmax=maximum)
                plt.xticks([])
                plt.yticks([])
                plt.title('True')
                colorbar = plt.colorbar(ticks=[0, maximum], fraction=0.05, aspect=10)
                colorbar.ax.tick_params(labelsize=6)

                plt.tight_layout()
                plt.show()

    return results


if __name__ == '__main__':
    dataset = CantileverDataset(
        normalize_inputs=False,
        transformation_exponentiation=None,
        transformation_logarithmic=None,
        label_max=100,
    )

    main(
        filename_model = 'StressNet.pth',
        train_existing = True,
        save_model_every = 5,
        save_best_separately = True,

        epoch_count = 50,
        learning_rate = 1e-3,
        decay_learning_rate = not True,
        batch_sizes = (16, 128, 128),
        dataset_split = (0.8, 0.1, 0.1),
        model = StressNet(2, 1, 32),
        Optimizer = torch.optim.Adam,
        Loss = nn.MSELoss,
        
        dataset = dataset,
        
        train = not True,
        test = True,
        show_loss = True,
        show_parity = True,
        show_predictions = True,
    )