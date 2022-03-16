'''
Train a CNN that predicts the stress contour in a cantilever beam subjected to a point load at the free end, with the following parameters that can vary:
- Geometry (length and height)
- Elastic modulus (constant throughout cantilever)
- Load magnitude
- Load direction
- Boundary conditions (fixed support)

The input images used to store these parameters contain four channels:
1. Load magnitude and direction: A black background with a grayscale line located at the location of the load. The brightness of the line represents the load magnitude, and the orientation of the line represents the load angle. Angle values follow the standard coordinate system and increase counterclockwise. An angle of 0 degrees points directly right, and an angle of 90 degrees points directly up.
2. Geometry: A black background with a white region representing the shape of the cantilever.
3. Elastic modulus: A black background with a grayscale region representing the elastic modulus values. The brightness of the region represents the magnitude of the elastic modulus.
4. Boundary conditions: A black background with a white line located at the location of the fixed support.

The output images used as labels during training are stress contours generated by FEA.
- Software: ANSYS (Mechanical APDL)
- Element type: PLANE182 (2D 4-node structural solid)
    - Thickness: 1
- Material properties:
    - Poisson's ratio: 0.3
'''


import glob
import os

from cycler import cycler
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
LEARNING_RATE = 0.0001  # 0.000001 for Nie
EPOCHS = 10
Model = FullyCnn


class CantileverDataset(Dataset):
    """Dataset that gets input and label images during training."""

    maxima = (None, None)

    def __init__(self, is_train=False):
        # Create input images and store each in a list.
        samples = read_samples(FILENAME_SAMPLES_TRAIN if is_train else FILENAME_SAMPLES_TEST)
        self.number_samples = get_sample_size(samples)
        self.inputs = generate_input_images(samples)
        # self.inputs = [np.zeros(INPUT_SIZE)] * self.number_samples  # Blank arrays to reduce startup time for debugging
        
        # Create label images from FEA stress data.
        self.labels, maxima = generate_label_images(
            samples,
            FOLDER_TRAIN_OUTPUTS if is_train else FOLDER_TEST_OUTPUTS,
            normalize=False,
            normalization_values=CantileverDataset.maxima,  #(1100000, None),  # Manually selected value,
            clip_high_stresses=False,
            )
        # self.labels, maxima = [np.zeros(OUTPUT_SIZE)] * self.number_samples, 0  # Blank arrays to reduce startup time for debugging
        # Store the maximum values found in the training dataset as a class variable to be referenced by the test datset.
        if is_train:
            CantileverDataset.maxima = maxima
        # # Write FEA images.
        # for i, label in enumerate(self.labels):
        #     with Image.fromarray((array_to_colormap(label[0, :, :])).astype(np.uint8)) as image:
        #         filepath = os.path.join(FOLDER_TRAIN_OUTPUTS, f'fea_{i+1}.png')
        #         image.save(filepath)

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

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
            # # Resize the output to the correct size.
            # output = nn.functional.interpolate(output, size=OUTPUT_SIZE[1:])
            test_loss += loss_function(output, label.float())

    test_loss /= batch_count
    print(f'Average loss: {test_loss:>8f}')
    return test_loss

def save(model):
    """Save model parameters to a file."""
    torch.save(model.state_dict(), FILEPATH_MODEL)
    print(f'Saved model parameters to {FILEPATH_MODEL}.')


# Train and test the model.
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Initialize the model and load its parameters if it has already been trained.
    model = Model()
    model.to(device)
    train_model = True
    if os.path.exists(FILEPATH_MODEL):
        model.load_state_dict(torch.load(FILEPATH_MODEL, map_location=torch.device(device)))
        model.eval()
        print(f'Loaded previously trained parameters from {FILEPATH_MODEL}.')
        if not input('Continue training this model? (y/n) ').lower().startswith('y'):
            train_model = False
    
    # Set up the training data.
    train_dataset = CantileverDataset(is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    if train_model:
        # Train the model and record the accuracy and loss.
        test_loss_values = []
        for t in range(EPOCHS):
            print(f'Epoch {t+1}\n------------------------')
            train(train_dataloader, model, loss_function, optimizer)
            test_loss = test(train_dataloader, model, loss_function)
            test_loss_values.append(test_loss)
            # Save the model parameters periodically.
            if (t+1) % 5 == 0:
                save(model)
        
        # Save the model parameters.
        save(model)
        
        # Plot the loss history.
        plt.figure()
        plt.plot(test_loss_values, '-o', color='#0095ff')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(axis='y')
        plt.show()

    # Set up the testing data.
    test_dataset = CantileverDataset(is_train=False)
    test_dataloader = DataLoader(test_dataset, shuffle=True)
    
    # Remove existing output images in the folder.
    filenames = glob.glob(os.path.join(FOLDER_TEST_OUTPUTS, '*.png'))
    for filename in filenames:
        os.remove(filename)
    print(f'Deleted {len(filenames)} existing images in {FOLDER_TEST_OUTPUTS}.')
    
    # Test the model on the input images found in the folder.
    evaluation_results = [{} for channel in range(OUTPUT_CHANNELS)]
    for i, (test_input, label) in enumerate(test_dataloader):
        test_input = test_input.to(device)
        label = label.to(device)
        test_output = model(test_input)
        test_output = test_output[0, :, ...].cpu().detach().numpy()
        label = label[0, :, ...].cpu().numpy()
        
        # The maximum stress found in the dataset, a hardcoded value that changes if a new dataset is generated.
        MAX_STRESS = 27620
        for channel in range(OUTPUT_CHANNELS):
            channel_name = ('stress', 'displacement')[channel]
            # Write FEA images.
            write_image(
                array_to_colormap(label[channel, ...], MAX_STRESS),
                os.path.join(FOLDER_TEST_OUTPUTS, f'{i+1}_fea_{channel_name}.png'),
                )
            # Evaluate outputs with multiple evaluation metrics.
            evaluation_result = metrics.evaluate(test_output[channel, ...], label[channel, ...])
            for name, result in evaluation_result.items():
                try:
                    evaluation_results[channel][name].append(result)
                except KeyError:
                    evaluation_results[channel][name] = [result]
            # Save the output image.
            write_image(
                array_to_colormap(test_output[channel, ...], MAX_STRESS),
                os.path.join(FOLDER_TEST_OUTPUTS, f'{i+1}_test_{channel_name}.png'),
                )
    print(f'Wrote {len(test_dataloader)} output images and {len(test_dataloader)} corresponding labels in {FOLDER_TEST_OUTPUTS}.')

    # Plot evaluation metrics.
    for channel in range(OUTPUT_CHANNELS):
        plt.rc('axes', prop_cycle=cycler(color=['#0095ff', '#ff4040']))
        plt.rc('font', family='Source Code Pro', size=12.0, weight='semibold')
        figure = plt.figure()
        for i, (name, result) in enumerate(evaluation_results[channel].items()):
            plt.subplot(1, len(evaluation_results[channel]), i+1)
            plt.plot(result, '.', markeredgewidth=5, label=['CNN', 'FEA'])
            if isinstance(result[0], tuple):
                plt.legend()
            plt.grid(visible=True, axis='y')
            plt.xlabel('Sample')
            plt.xticks(range(len(test_dataset)))
            # plt.ylim((0, 1))
            plt.title(name, fontweight='bold')
        plt.show()