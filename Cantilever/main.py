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

import metrics
from networks import *
from setup import *


# Model parameters file name and path.
FILEPATH_MODEL = os.path.join(FOLDER_ROOT, 'model.pth')
# Training hyperparameters.
BATCH_SIZE = 1
LEARNING_RATE = 0.00001  # 0.000001 for Nie
EPOCHS = 50
Model = FullyCnn


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

def train(dataloader, model, loss_function, optimizer):
    """Train the model for one epoch only."""

    for batch, (data, label) in enumerate(dataloader):
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
        if batch % 10 == 0:
            print(f'Batch {batch}...', end='\r')
    print()

def test(dataloader, model, loss_function):
    """Test the model for one epoch only."""

    batch_count = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            test_loss += loss_function(output, label.float())

    test_loss /= batch_count
    print(f'Average loss: {test_loss:>8f}')
    return test_loss

def save(epoch, model, optimizer, loss):
    """Save model parameters to a file."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, FILEPATH_MODEL)
    print(f'Saved model parameters to {FILEPATH_MODEL}.')


# Train and test the model.
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Initialize the model and optimizer and load their parameters if they have been saved previously.
    model = Model()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()
    
    train_model = True
    epochs = range(EPOCHS)
    if os.path.exists(FILEPATH_MODEL):
        checkpoint = torch.load(FILEPATH_MODEL, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        epochs = range(epoch, epoch+EPOCHS)
        loss = checkpoint['loss']

        print(f'Loaded previously trained parameters from {FILEPATH_MODEL}.')
        if not input('Continue training this model? (y/n) ').lower().startswith('y'):
            train_model = False
        
        model.train(train_model)
    
    # Set up the training and validation data.
    DESIRED_SAMPLE_SIZE = 390
    samples = read_samples(FILENAME_SAMPLES_TRAIN)
    samples = get_stratified_samples(samples, FOLDER_TRAIN_LABELS, bins=10, 
    desired_sample_size=DESIRED_SAMPLE_SIZE)

    sample_size_train = int(0.8 * DESIRED_SAMPLE_SIZE)
    train_samples = {key: value[:sample_size_train] for key, value in samples.items()}
    validation_samples = {key: value[sample_size_train:] for key, value in samples.items()}

    train_dataset = CantileverDataset(train_samples, FOLDER_TRAIN_LABELS)
    validation_dataset = CantileverDataset(validation_samples, FOLDER_TRAIN_LABELS)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if train_model:
        # Train the model and record the accuracy and loss.
        test_loss_values = []
        for epoch in epochs:
            print(f'Epoch {epoch+1}\n------------------------')
            train(train_dataloader, model, loss_function, optimizer)
            test_loss = test(validation_dataloader, model, loss_function)
            test_loss_values.append(test_loss)
            # Save the model parameters periodically.
            if (epoch+1) % 5 == 0:
                save(epoch, model, optimizer, test_loss)
        
        # Save the model parameters.
        save(epoch, model, optimizer, test_loss)
        
        # Plot the loss history.
        plt.figure()
        plt.plot(epochs, test_loss_values, '-o', color='#0095ff')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(axis='y')
        plt.show()

    # Set up the testing data.
    test_samples = read_samples(FILENAME_SAMPLES_TEST)
    test_dataset = CantileverDataset(test_samples, FOLDER_TEST_LABELS)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    
    # Remove existing output images in the folder.
    # filenames = glob.glob(os.path.join(FOLDER_TEST_LABELS, '*.png'))
    # for filename in filenames:
    #     os.remove(filename)
    # print(f'Deleted {len(filenames)} existing images in {FOLDER_TEST_LABELS}.')
    
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
    # evaluation_results = [{} for channel in range(OUTPUT_CHANNELS)]
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
            # # Evaluate outputs with multiple evaluation metrics.
            # evaluation_result = metrics.evaluate(test_output[channel, ...], label[channel, ...])
            # for name, result in evaluation_result.items():
            #     try:
            #         evaluation_results[channel][name].append(result)
            #     except KeyError:
            #         evaluation_results[channel][name] = [result]
    print(f'Wrote {len(test_dataloader)} output images and {len(test_dataloader)} corresponding labels in {FOLDER_ROOT}.')

    # Calculate and plot evaluation metrics.
    BLUE = '#0095ff'
    RED = '#ff4040'
    for channel, channel_name in enumerate(OUTPUT_CHANNEL_NAMES):
        # plt.rc('font', family='Source Code Pro', size=10.0, weight='semibold')

        # Area metric.
        figure = plt.figure()
        NUMBER_COLUMNS = 4
        for i, (test_output, test_label) in enumerate(zip(test_outputs, test_labels)):
            plt.subplot(math.ceil(len(test_outputs) / NUMBER_COLUMNS), NUMBER_COLUMNS, i+1)
            cdf_network, cdf_label, bins, area_difference = metrics.area_metric(test_output[channel, ...], test_label[channel, ...], max_values[channel])
            plt.plot(bins[1:], cdf_network, '-', color=BLUE, label='CNN')
            plt.plot(bins[1:], cdf_label, '--', color=RED, label='FEA')
            plt.legend()
            plt.grid(visible=True, axis='y')
            plt.yticks([0, 1])
            plt.title(f"[#{i+1}] {area_difference:0.2f}", fontsize=10, fontweight='bold')
        plt.suptitle(f"Area Metric ({channel_name.capitalize()})", fontweight='bold')
        plt.tight_layout()  # Increase spacing between subplots
        plt.show()

        # Mean error.
        figure = plt.figure()
        me = []
        sample_numbers = range(1, len(test_outputs)+1)
        for i, (test_output, test_label) in enumerate(zip(test_outputs, test_labels)):
            me.append(
                metrics.mean_error(test_output[channel, ...], test_label[channel, ...])
            )
        plt.plot(sample_numbers, me, '.', markeredgewidth=5, color=BLUE)
        plt.grid()
        plt.xlabel("Sample Number")
        plt.xticks(sample_numbers)
        plt.ylabel("Mean Error")
        plt.title(f"Mean Error ({channel_name.capitalize()})", fontweight='bold')
        plt.show()

        # for i, (name, result) in enumerate(evaluation_results[channel].items()):
        #     plt.subplot(1, len(evaluation_results[channel]), i+1)
        #     plt.plot(result, '.', markeredgewidth=5, label=['CNN', 'FEA'])
        #     if isinstance(result[0], tuple):
        #         plt.legend()
        #     plt.grid(visible=True, axis='y')
        #     plt.xlabel('Sample')
        #     plt.xticks(range(len(test_dataset)))
        #     # plt.ylim((0, 1))
        #     plt.title(name, fontweight='bold')
        # plt.show()