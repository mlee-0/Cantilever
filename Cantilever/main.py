'''
Train and test the model.
'''


import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from datasets import *
import metrics
from networks import *
from setup import *


# Model parameters file path.
FILEPATH_MODEL = os.path.join(FOLDER_ROOT, 'model.pth')


class CantileverDataset(Dataset):
    """Dataset that gets input and label images during training."""
    def __init__(self, samples: dict, folder_labels):
        self.number_samples = get_sample_size(samples)
        # Create label images.
        self.labels = generate_label_images(
            samples,
            folder_labels,
            )
        print(f"Label images take up {sys.getsizeof(self.labels)/1e9:,.2f} GB.")
        # Create input images.
        self.inputs = generate_input_images(samples)
        print(f"Input images take up {sys.getsizeof(self.inputs)/1e9:,.2f} GB.")

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        # Return copies of arrays so that arrays are not modified.
        return np.copy(self.inputs[index, ...]), np.copy(self.labels[index, ...])

def save(epoch, model, optimizer, loss_history) -> None:
    """Save model parameters to a file."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history,
    }, FILEPATH_MODEL)
    print(f'Saved model parameters to {FILEPATH_MODEL}.')

def main(epoch_count: int, learning_rate: float, batch_size: int, desired_subset_size: int, bins: int, nonuniformity: float, training_split: float, Model: nn.Module, keep_training=None, test_only=False, queue=None, queue_to_main=None):
    """
    Train and test the model.

    `desired_subset_size`: Number of samples to include in the subset. Enter 0 to use all samples found instead of creating a subset.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Initialize the model and optimizer and load their parameters if they have been saved previously.
    model = Model()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    epochs = range(1, epoch_count+1)
    previous_test_loss = []
    if os.path.exists(FILEPATH_MODEL):
        if not test_only:
            if keep_training is None:
                keep_training = input(f'Continue training the model in {FILEPATH_MODEL}? [y/n] ') == 'y'
        else:
            keep_training = True
        
        if keep_training:
            checkpoint = torch.load(FILEPATH_MODEL, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1
            epochs = range(epoch, epoch+epoch_count)
            previous_test_loss = checkpoint['loss']
    else:
        keep_training = False
        test_only = False
    
    # Create a subset of the entire dataset, or load the previously created subset.
    samples = read_samples(FILENAME_SAMPLES_TRAIN)
    if desired_subset_size > 0:
        filename_subset = os.path.join(FOLDER_ROOT, FILENAME_SUBSET)
        try:
            with open(filename_subset, 'r') as f:
                sample_numbers = [int(_) for _ in f.readlines()]
            sample_indices = np.array(sample_numbers) - 1
            samples = {key: [value[i] for i in sample_indices] for key, value in samples.items()}
            print(f"Using previously created subset with {len(sample_numbers)} samples from {filename_subset}.")
        except FileNotFoundError:
            samples = get_stratified_samples(samples, FOLDER_TRAIN_LABELS, 
            desired_subset_size=desired_subset_size, bins=bins, nonuniformity=nonuniformity)
            with open(filename_subset, 'w') as f:
                f.writelines([f"{_}\n" for _ in samples[KEY_SAMPLE_NUMBER]])
            print(f"Wrote subset with {len(samples[KEY_SAMPLE_NUMBER])} samples to {filename_subset}.")
    sample_size = get_sample_size(samples)

    # Set up the training and validation data.
    sample_size_train, sample_size_validation = split_training_validation(sample_size, training_split)
    train_samples = {key: value[:sample_size_train] for key, value in samples.items()}
    validation_samples = {key: value[sample_size_train:sample_size_train+sample_size_validation:] for key, value in samples.items()}
    
    # plt.figure()
    # plt.subplot(1, 4, 1)
    # plt.hist(validation_samples["Load"], 20)
    # plt.subplot(1, 4, 2)
    # plt.hist(validation_samples["Angle"], 20, (0, 360))
    # plt.subplot(1, 4, 3)
    # plt.hist(validation_samples["Length"], 20, (2, 4))
    # plt.subplot(1, 4, 4)
    # plt.hist(validation_samples["Height"], 20, (1, 2))
    # plt.show()

    train_dataset = CantileverDataset(train_samples, FOLDER_TRAIN_LABELS)
    validation_dataset = CantileverDataset(validation_samples, FOLDER_TRAIN_LABELS)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    size_train_dataset = len(train_dataloader)
    size_validation_dataset = len(validation_dataloader)
    print(f"Split {len(samples[KEY_SAMPLE_NUMBER])} samples into {size_train_dataset} training / {size_validation_dataset} validation.")

    if not test_only:
        if queue:
            queue.put([(epochs[0]-1, epochs[-1]), None, None, None, None])

        validation_loss = []
        for epoch in epochs:
            print(f'Epoch {epoch}\n------------------------')
            
            # Train on the training dataset.
            model.train(True)
            for batch, (data, label) in enumerate(train_dataloader, 1):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = loss_function(output, label.float())
                # Reset gradients of model parameters.
                optimizer.zero_grad()
                # Backpropagate the prediction loss.
                loss.backward()
                # Adjust model parameters.
                optimizer.step()

                if (batch) % 100 == 0:
                    print(f"Training batch {batch}/{size_train_dataset} with loss {loss:,.0f}...", end="\r")
                    if queue:
                        queue.put([None, (batch, size_train_dataset+size_validation_dataset), None, None, None])
            print()

            # Train on the validation dataset. Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
            model.train(False)
            loss = 0
            with torch.no_grad():
                for batch, (data, label) in enumerate(validation_dataloader, 1):
                    data = data.to(device)
                    label = label.to(device)
                    output = model(data)
                    loss += loss_function(output, label.float())
                    if (batch) % 100 == 0:
                        print(f"Validating batch {batch}/{size_validation_dataset}...", end="\r")
                        if queue:
                            queue.put([None, (size_train_dataset+batch, size_train_dataset+size_validation_dataset), None, None, None])
            print()
            loss /= size_validation_dataset
            validation_loss.append(loss)
            print(f"Average loss: {loss:,.0f}")

            # Save the model parameters periodically.
            if (epoch) % 5 == 0:
                save(epoch, model, optimizer, [*previous_test_loss, *validation_loss])
            
            if queue:
                queue.put([(epoch, epochs[-1]), None, epochs, validation_loss, previous_test_loss])
            
            if queue_to_main:
                if not queue_to_main.empty():
                    # Stop training.
                    if queue_to_main.get() == True:
                        queue_to_main.queue.clear()
                        break
        
        # Save the model parameters.
        save(epoch, model, optimizer, [*previous_test_loss, *validation_loss])
        
        # Plot the loss history.
        if not queue:
            plt.figure()
            if previous_test_loss:
                plt.plot(range(1, epochs[0]), previous_test_loss, 'o', color=Colors.GRAY_LIGHT)
            plt.plot(epochs, validation_loss, '-o', color=Colors.BLUE)
            plt.ylim(bottom=0)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(axis='y')
            plt.show()

    # Set up the testing data.
    test_samples = read_samples(FILENAME_SAMPLES_TEST)
    test_dataset = CantileverDataset(test_samples, FOLDER_TEST_LABELS)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    size_test_dataset = len(test_dataloader)
    
    # The maximum values found among the training and testing datasets for each channel. Used to normalize values for images.
    max_values = [
        max([
            np.max(train_dataset.labels[:, channel, ...]),
            np.max(test_dataset.labels[:, channel, ...]),
        ])
        for channel in range(OUTPUT_CHANNELS)
    ]
    # Test on the testing dataset.
    model.train(False)
    test_labels = []
    test_outputs = []
    with torch.no_grad():
    for batch, (test_input, label) in enumerate(test_dataloader, 1):
        test_input = test_input.to(device)
        label = label.to(device)
        test_output = model(test_input)
        test_output = test_output[0, :, ...].cpu().detach().numpy()
        label = label[0, :, ...].cpu().numpy()
        test_labels.append(label)
        test_outputs.append(test_output)
        
        for channel, channel_name in enumerate(OUTPUT_CHANNEL_NAMES):
            # Write the combined FEA and model output image.
            image = np.vstack((
                label[channel, ...],  # FEA
                test_output[channel, ...],  # Model output
            ))
            write_image(
                array_to_colormap(image, max_values[channel] if channel_name == "stress" else None),
                os.path.join(FOLDER_ROOT, f'{batch}_fea_model_{channel_name}.png'),
                )
        
        if queue:
                queue.put([None, (batch, size_test_dataset), None, None, None])
    
    print(f"Wrote {size_test_dataset} test images in {FOLDER_ROOT}.")

    # Calculate and plot evaluation metrics.
    for channel, channel_name in enumerate(OUTPUT_CHANNEL_NAMES):
        if queue:
            break

        # plt.rc('font', family='Source Code Pro', size=10.0, weight='semibold')

        # Area metric.
        plt.figure()
        NUMBER_COLUMNS = 4
        for i, (test_output, test_label) in enumerate(zip(test_outputs, test_labels)):
            plt.subplot(math.ceil(len(test_outputs) / NUMBER_COLUMNS), NUMBER_COLUMNS, i+1)
            cdf_network, cdf_label, bin_edges, area_difference = metrics.area_metric(test_output[channel, ...], test_label[channel, ...], max_values[channel])
            plt.plot(bin_edges[1:], cdf_network, '-', color=Colors.BLUE)
            plt.plot(bin_edges[1:], cdf_label, '--', color=Colors.RED)
            if i == 0:
                plt.legend(["CNN", "FEA"])
            plt.grid(visible=True, axis='y')
            plt.xticks([0, max_values[channel]])
            plt.yticks([0, 1])
            plt.title(f"[#{i+1}] {area_difference:0.2f}", fontsize=10, fontweight='bold')
        plt.suptitle(f"Area Metric ({channel_name.capitalize()})", fontweight='bold')
        plt.tight_layout()  # Increase spacing between subplots
        plt.show()

        # Single-value error metrics.
        mv, me, mae, mse, mre = [], [], [], [], []
        results = {"Maximum Value": mv, "Mean Error": me, "Mean Absolute Error": mae, "Mean Squared Error": mse, "Mean Relative Error": mre}
        for test_output, test_label in zip(test_outputs, test_labels):
            test_output, test_label = test_output[channel, ...], test_label[channel, ...]
            mv.append(metrics.maximum_value(test_output, test_label))
            me.append(metrics.mean_error(test_output, test_label))
            mae.append(metrics.mean_absolute_error(test_output, test_label))
            mse.append(metrics.mean_squared_error(test_output, test_label))
            mre.append(metrics.mean_relative_error(test_output, test_label))
        
        sample_numbers = range(1, len(test_outputs)+1)
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
            plt.xticks(sample_numbers)
            plt.title(metric)
        plt.suptitle(f"{channel_name.capitalize()}", fontweight='bold')
        plt.show()


if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS = 1
    LEARNING_RATE = 1e-7  #0.00001  # 0.000001 for Nie
    BATCH_SIZE = 1
    Model = Nie
    DESIRED_SAMPLE_SIZE = 10000
    BINS = 1
    NON_UNIFORMITY = 1.0
    TRAINING_SPLIT = 0.8

    main(
        EPOCHS, LEARNING_RATE, BATCH_SIZE, DESIRED_SAMPLE_SIZE, BINS, NON_UNIFORMITY, TRAINING_SPLIT, Model,
        keep_training=True, test_only=False,
    )