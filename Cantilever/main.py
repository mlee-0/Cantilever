'''
Train a CNN that predicts the stress contour in a fixed-size cantilever beam with a constant-magnitude point load at the end that varies in direction.

Properties of the cantilever beam:
    - Dimensions: 1 m (length), 0.25 m (height), 0.25 m (depth).
    - Material: structural steel.
    - Elastic modulus: 200 GPa.

The input images are black-and-white binary images with a white line oriented at the specific angle. Angle values follow the standard coordinate system and increase counterclockwise. An angle of 0 degrees points directly right, and an angle of 90 degrees points directly up. The output images are stress contours in the cantilever.
'''


import colorsys
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


device = 'cpu'  #'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')


# Size of input images.
INPUT_SIZE = (100, 100)
# Size to which output images are resized to account for different output image sizes.
OUTPUT_SIZE = (100, 25)

# Folders.
FOLDER_ROOT = 'Cantilever'
FOLDER_TRAIN_INPUTS = os.path.join(FOLDER_ROOT, 'Train Inputs')
FOLDER_TRAIN_OUTPUTS = os.path.join(FOLDER_ROOT, 'Train Outputs')
FOLDER_TEST_INPUTS = os.path.join(FOLDER_ROOT, 'Test Inputs')
FOLDER_TEST_OUTPUTS = os.path.join(FOLDER_ROOT, 'Test Outputs')

# Training hyperparameters.
BATCH_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 30


# Create images for the sample values provided inside the specified folder.
def generate_input_images(angle_samples, folder_inputs):
    # Remove existing input images in the folder.
    filenames = glob.glob(os.path.join(folder_inputs, '*.png'))
    for filename in filenames:
        os.remove(filename)
    print(f'Deleted {len(filenames)} existing images in {folder_inputs}.')

    filenames = []
    for angle in angle_samples:
        angle = int(angle)
        # Create a black image and add a white line of pixels representing the load direction.
        image = np.zeros(INPUT_SIZE)
        r = np.arange(INPUT_SIZE[0])
        x = r * np.cos(angle * np.pi/180) + INPUT_SIZE[1]/2
        y = r * np.sin(angle * np.pi/180) + INPUT_SIZE[0]/2
        x = x.astype(int)
        y = y.astype(int)
        valid_indices = (x >= 0) * (x < INPUT_SIZE[0]) * (y >= 0) * (y < INPUT_SIZE[1])
        image[y[valid_indices], x[valid_indices]] = 255
        image = np.flipud(image)
        # Write image files.
        filename = os.path.join(folder_inputs, f'input_{angle}.png')
        with Image.fromarray(image.astype(np.uint8), 'L').convert('1') as file:
            file.save(filename)
            filenames.append(filename)
    print(f'Wrote {len(angle_samples)} input images in {folder_inputs}.')
    return filenames

# Convert a 3-channel RGB array into a 1-channel hue array with values in [0, 1].
def rgb_to_hue(array):
    array = array / 255
    hue_array = np.zeros((array.shape[0], array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            hsv = colorsys.rgb_to_hsv(array[i, j, 0], array[i, j, 1], array[i, j, 2])
            hue_array[i, j] = hsv[0]
    return hue_array

# Convert a 3-channel HSV array into a 3-channel RGB array.
def hsv_to_rgb(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            rgb = colorsys.hsv_to_rgb(array[i, j, 0], array[i, j, 1], array[i, j, 2])
            for k in range(3):
                array[i, j, k] = rgb[k] * 255
    return array

# Dataset that retrieves input and output images for the CNN. Output images are not retrieved if the outputs folder is not specified.
class CantileverDataset(Dataset):
    def __init__(self, folder_inputs, folder_outputs=None):
        self.folder_inputs = folder_inputs
        self.folder_outputs = folder_outputs
        # Get all input image filenames in the inputs folder.
        self.filenames = glob.glob(os.path.join(self.folder_inputs, '*.png'))
        self.filename_suffixes = [filename.split('.')[-2].split('_')[1:] for filename in self.filenames]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename_input = self.filenames[index]
        image_input = np.asarray(Image.open(filename_input), np.uint8)
        image_input = np.expand_dims(image_input, 0)
        if self.folder_outputs is not None:
            filename_output = os.path.join(
                self.folder_outputs,
                f'{"_".join(["stress"] + self.filename_suffixes[index])}.png',
                )
            image_output = np.asarray(Image.open(filename_output), np.uint8)[:, :, :3]
            image_output = rgb_to_hue(image_output).flatten()
        else:
            image_output = 0
        return image_input, image_output

# A CNN that predicts the stress contour in a cantilever beam with a point load at the free end.
class StressContourNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.linear = nn.Sequential(
            nn.Linear(in_features=2116, out_features=OUTPUT_SIZE[0]*OUTPUT_SIZE[1]),
        )
    
    def forward(self, x):
        x = x.float()
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# Train the model for one epoch only.
def train(dataloader, model, loss_function, optimizer):
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

        if batch % (BATCH_SIZE * 10) == 0:
            loss, current = loss.item(), batch * len(data)
            print(f'Loss: {loss:>7f}  (batch {current} of {len(dataloader.dataset)})')

# Test the model for one epoch only.
def test(dataloader, model, loss_function):
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


# Train and test the model.
if __name__ == '__main__':
    # Initialize the model.
    model = StressContourNetwork()
    
    # Set up the training data.
    train_dataset = CantileverDataset(FOLDER_TRAIN_INPUTS, FOLDER_TRAIN_OUTPUTS)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Train the model and record the accuracy and loss.
    test_loss_values = []
    for t in range(EPOCHS):
        print(f'Epoch {t+1}\n------------------------')
        train(train_dataloader, model, loss_function, optimizer)
        test_loss = test(train_dataloader, model, loss_function)
        test_loss_values.append(test_loss)
    
    # Plot the loss history.
    plt.figure()
    plt.plot(test_loss_values, '-o', color='#ffbf00')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(axis='y')
    plt.show()

    # Set up the testing data.
    test_dataset = CantileverDataset(FOLDER_TEST_INPUTS, None)
    test_dataloader = DataLoader(test_dataset, shuffle=True)
    
    # Remove existing output images in the folder.
    filenames = glob.glob(os.path.join(FOLDER_TEST_OUTPUTS, '*.png'))
    for filename in filenames:
        os.remove(filename)
    print(f'Deleted {len(filenames)} existing images in {FOLDER_TEST_OUTPUTS}.')
    
    # Test the model on the input images found in the folder.
    for index, (test_input, _) in enumerate(test_dataloader):
        test_output = model(test_input).detach().numpy()[0, :]
        test_output = test_output.reshape(OUTPUT_SIZE, order='F')
        # Convert the output to an RGB array.
        test_output = np.dstack((test_output, np.ones(OUTPUT_SIZE, float), 2/3 * np.ones(OUTPUT_SIZE, float)))
        test_output = hsv_to_rgb(test_output)
        # Save the generated output image.
        test_output = test_output.transpose((1, 0, 2)).astype(np.uint8)
        with Image.fromarray(test_output) as image:
            image.save(os.path.join(
                FOLDER_TEST_OUTPUTS,
                f'{"_".join(["test"] + test_dataset.filename_suffixes[index])}.png',
                ))
    print(f'Wrote {len(test_dataloader)} output images in {FOLDER_TEST_OUTPUTS}.')