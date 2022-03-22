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
        # Create input images.
        self.number_samples = get_sample_size(samples)
        self.inputs = generate_input_images(samples)
        # self.inputs = np.zeros((self.number_samples, *INPUT_SIZE))  # Blank arrays to reduce startup time for debugging
        
        # Create label images.
        self.labels = generate_label_images(
            samples,
            folder_labels,
            )
        # self.labels = np.zeros((self.number_samples, *OUTPUT_SIZE))  # Blank arrays to reduce startup time for debugging
        # # Write FEA images.
        # for i, label in enumerate(self.labels):
        #     with Image.fromarray((array_to_colormap(label[0, :, :])).astype(np.uint8)) as image:
        #         filepath = os.path.join(FOLDER_TRAIN_LABELS, f'fea_{i+1}.png')
        #         image.save(filepath)

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        # Return copies of arrays so that arrays are not modified.
        return np.copy(self.inputs[index, ...]), np.copy(self.labels[index, ...])

def save(epoch, model, optimizer, loss_history):
    """Save model parameters to a file."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history,
    }, FILEPATH_MODEL)
    print(f'Saved model parameters to {FILEPATH_MODEL}.')

def main(epoch_count: int, learning_rate: float, batch_size: int, desired_sample_size: int, bins: int, training_split: float, Model: nn.Module, keep_training=None, test_only=False, queue=None, queue_to_main=None):
    """Train and test the model."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Initialize the model and optimizer and load their parameters if they have been saved previously.
    model = Model()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    epochs = range(epoch_count)
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
            
            model.train(not test_only)
    else:
        keep_training = False
        test_only = False
    
    # Create a subset of the entire dataset, or load the previously created subset.
    samples = read_samples(FILENAME_SAMPLES_TRAIN)
    filename_subset = os.path.join(FOLDER_ROOT, FILENAME_SUBSET)
    try:
        with open(filename_subset, 'r') as f:
            sample_numbers = [int(_) for _ in f.readlines()]
        sample_indices = np.array(sample_numbers) - 1
        samples = {key: [value[i] for i in sample_indices] for key, value in samples.items()}
        print(f"Using previously created subset from {filename_subset}, with {len(sample_numbers)} samples.")
    except FileNotFoundError:
        samples = get_stratified_samples(samples, FOLDER_TRAIN_LABELS, bins=bins, 
        desired_subset_size=desired_sample_size)
        with open(filename_subset, 'w') as f:
            f.writelines([f"{_}\n" for _ in samples[KEY_SAMPLE_NUMBER]])

    # Set up the training and validation data.
    sample_size_train = round(training_split * desired_sample_size)
    train_samples = {key: value[:sample_size_train] for key, value in samples.items()}
    validation_samples = {key: value[sample_size_train:] for key, value in samples.items()}

    train_dataset = CantileverDataset(train_samples, FOLDER_TRAIN_LABELS)
    validation_dataset = CantileverDataset(validation_samples, FOLDER_TRAIN_LABELS)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    if not test_only:
        # Train the model and record the accuracy and loss.
        test_loss = []
        for epoch in epochs:
            print(f'Epoch {epoch+1}\n------------------------')
            
            # Train on the training dataset.
            for batch, (data, label) in enumerate(train_dataloader):
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
                # if batch % (BATCH_SIZE * 10) == 0:
                #     loss, current = loss.item(), batch * len(data)
                #     print(f'Loss: {loss:>7f}  (batch {current} of {len(dataloader.dataset)})')
                if (batch+1) % 10 == 0:
                    print(f'Batch {batch+1}...', end='\r')
            print()

            # Train on the validation dataset.
            batch_count = len(validation_dataloader)
            loss = 0
            with torch.no_grad():
                for data, label in validation_dataloader:
                    data = data.to(device)
                    label = label.to(device)
                    output = model(data)
                    loss += loss_function(output, label.float())
            loss /= batch_count
            test_loss.append(loss)
            print(f'Average loss: {loss:>8f}')

            # Save the model parameters periodically.
            if (epoch+1) % 5 == 0:
                save(epoch, model, optimizer, [*previous_test_loss, *test_loss])
            
            if queue:
                queue.put([epoch, epochs, test_loss, previous_test_loss])
            
            if queue_to_main:
                if not queue_to_main.empty():
                    # Stop training.
                    if queue_to_main.get() == True:
                        queue_to_main.queue.clear()
                        break
        
        # Save the model parameters.
        save(epoch, model, optimizer, [*previous_test_loss, *test_loss])
        
        # Plot the loss history.
        if not queue:
            plt.figure()
            if previous_test_loss:
                plt.plot(range(epochs[0]), previous_test_loss, 'o', color=Colors.GRAY)
            plt.plot(epochs, test_loss, '-o', color=Colors.BLUE)
            plt.ylim(bottom=0)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(axis='y')
            plt.show()

    # Set up the testing data.
    test_samples = read_samples(FILENAME_SAMPLES_TEST)
    test_dataset = CantileverDataset(test_samples, FOLDER_TEST_LABELS)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    
    # The maximum values found among the training and testing datasets for each channel. Used to normalize values for images.
    max_values = [
        max([
            np.max(train_dataset.labels[:, channel, ...]),
            np.max(test_dataset.labels[:, channel, ...]),
        ])
        for channel in range(OUTPUT_CHANNELS)
    ]
    # Test the model on the input images found in the folder.
    test_labels = []
    test_outputs = []
    for i, (test_input, label) in enumerate(test_dataloader):
        test_input = test_input.to(device)
        label = label.to(device)
        test_output = model(test_input)
        test_output = test_output[0, :, ...].cpu().detach().numpy()
        label = label[0, :, ...].cpu().numpy()
        test_labels.append(label)
        test_outputs.append(test_output)
        
        for channel, channel_name in enumerate(OUTPUT_CHANNEL_NAMES):
            # Write the FEA image.
            write_image(
                array_to_colormap(label[channel, ...], max_values[channel] if channel_name == "stress" else None),
                os.path.join(FOLDER_ROOT, f'{i+1}_fea_{channel_name}.png'),
                )
            # Write the output image.
            write_image(
                array_to_colormap(test_output[channel, ...], max_values[channel]),
                os.path.join(FOLDER_ROOT, f'{i+1}_test_{channel_name}.png'),
                )
    print(f'Wrote {len(test_dataloader)} output images and {len(test_dataloader)} corresponding labels in {FOLDER_ROOT}.')

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
            cdf_network, cdf_label, bins, area_difference = metrics.area_metric(test_output[channel, ...], test_label[channel, ...], max_values[channel])
            plt.plot(bins[1:], cdf_network, '-', color=Colors.BLUE, label='CNN')
            plt.plot(bins[1:], cdf_label, '--', color=Colors.RED, label='FEA')
            plt.legend()
            plt.grid(visible=True, axis='y')
            plt.yticks([0, 1])
            plt.title(f"[#{i+1}] {area_difference:0.2f}", fontsize=10, fontweight='bold')
        plt.suptitle(f"Area Metric ({channel_name.capitalize()})", fontweight='bold')
        plt.tight_layout()  # Increase spacing between subplots
        plt.show()

        # Mean error.
        plt.figure()
        me = []
        sample_numbers = range(1, len(test_outputs)+1)
        for i, (test_output, test_label) in enumerate(zip(test_outputs, test_labels)):
            me.append(
                metrics.mean_error(test_output[channel, ...], test_label[channel, ...])
            )
        plt.plot(sample_numbers, me, '.', markeredgewidth=5, color=Colors.BLUE)
        plt.grid()
        plt.xlabel("Sample Number")
        plt.xticks(sample_numbers)
        plt.ylabel("Mean Error")
        plt.title(f"Mean Error ({channel_name.capitalize()})", fontweight='bold')
        plt.show()


if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS = 10
    LEARNING_RATE = 0.00001  # 0.000001 for Nie
    BATCH_SIZE = 1
    Model = FullyCnn
    DESIRED_SAMPLE_SIZE = 730
    BINS = 4 #10
    TRAINING_SPLIT = 0.8

    main(
        EPOCHS, LEARNING_RATE, BATCH_SIZE, DESIRED_SAMPLE_SIZE, BINS, TRAINING_SPLIT, Model,
        keep_training=True, test_only=False,
    )