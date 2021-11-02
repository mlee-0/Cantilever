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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')


# Dataset size.
NUMBER_SAMPLES = 20
# Load magnitude.
LOAD = 50000
# Minimum and maximum values of parameters to be varied.
ANGLE_RANGE = (0, 90)
# Increments used to generate samples within the above ranges.
ANGLE_STEP = 3
# Size of input images.
INPUT_SIZE = (100, 100)
# Size to which output images are resized to account for different output image sizes.
OUTPUT_SIZE = (100, 25)

# Folders.
FOLDER_ROOT = 'Cantilever'
FOLDER_INPUTS = os.path.join(FOLDER_ROOT, 'Inputs')
FOLDER_OUTPUTS = os.path.join(FOLDER_ROOT, 'Outputs')
# Files.
FILENAME_TEXT = 'cantilever_samples.txt'

# Training hyperparameters.
BATCH_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 30


# Return randomly generated sample values for load magnitudes and angles and write them to a text file.
def generate_samples():
    # Generate equally spaced values within the corresponding range, using the specified increments.
    angle_samples = np.arange(ANGLE_RANGE[0], ANGLE_RANGE[1], ANGLE_STEP)
    # Randomize the ordering of the samples.
    random.shuffle(angle_samples)
    # Keep only the first N samples.
    angle_samples = angle_samples[:NUMBER_SAMPLES]
    
    # Determine the x and y components of the load for easier entry in ANSYS.
    load_samples = []
    for angle in angle_samples:
        angle *= (np.pi / 180)
        load_samples.append((
            np.cos(angle) * LOAD,
            np.sin(angle) * LOAD,
            ))
    # Write samples to text file.
    text = [f'X load: {load[0]:.2f},  Y load: {load[1]:.2f},  Angle: {angle}\n' for load, angle in zip(load_samples, angle_samples)]
    with open(os.path.join(FOLDER_ROOT, FILENAME_TEXT), 'w') as file:
        file.writelines(text)
    return angle_samples

# Return the sample values found in the text file previously generated.
def read_samples():
    angle_samples = []
    filename = os.path.join(FOLDER_ROOT, FILENAME_TEXT)
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            for string in file.readlines():
                *_, angle = [int(float(string.split(':')[1])) for string in string.split(',')]
                angle_samples.append(angle)
    return angle_samples

# Create images for each pair of sample values provided.
def generate_input_images(angle_samples):
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
        filename = os.path.join(FOLDER_INPUTS, f'input_{angle}.png')
        with Image.fromarray(image.astype(np.uint8), 'L').convert('1') as file:
            file.save(filename)
            filenames.append(filename)
    return filenames

# Crop the stress contour region of the output images.
def crop_output_images():
    # LEFT, TOP = 209, 108
    # SIZE = (616, 155)
    filenames = glob.glob(os.path.join(FOLDER_OUTPUTS, '*.png'))
    for filename in filenames:
        with Image.open(filename) as image:
            # if image.size[0] > SIZE[0] and image.size[1] > SIZE[1]:
            #     image = image.crop((LEFT, TOP, LEFT+SIZE[0]-1, TOP+SIZE[1]-1))
            image_copy = image.convert('L')
            area = ImageOps.invert(image_copy).getbbox()
            image = image.crop(area)
            image = image.resize(OUTPUT_SIZE)
            image.save(filename)

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

class CantileverDataset(Dataset):
    def __init__(self, folder_inputs, folder_outputs):
        self.folder_inputs = folder_inputs
        self.folder_outputs = folder_outputs
        # Get all image filenames in the folder.
        self.filenames = glob.glob(os.path.join(folder_inputs, '*.png'))

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename_input = self.filenames[index]
        filename_output = os.path.join(FOLDER_OUTPUTS, '_'.join(['stress'] + self.filenames[index].split('_')[1:]))
        image_input = np.asarray(Image.open(filename_input), np.uint8)
        image_output = np.asarray(Image.open(filename_output), np.uint8)[:, :, :3]
        image_output = rgb_to_hue(image_output).flatten()
        image_input = np.expand_dims(image_input, 0)
        return image_input, image_output

# A CNN that predicts the stress contour in a fixed-size cantilever beam with a point load that varies in magnitude and direction.
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
            print(f'Loss: {loss:>7f}  (batch {batch * len(data)})')

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


# Try to read sample values from the text file if it already exists. If not, generate the samples.
angle_samples = read_samples()
if not angle_samples:
    angle_samples = generate_samples()
generate_input_images(angle_samples)
# Crop output images.
crop_output_images()

# Set up the data and model.
dataset = CantileverDataset(FOLDER_INPUTS, FOLDER_OUTPUTS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
size = len(dataloader.dataset)
model = StressContourNetwork()
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Train the model and record the accuracy and loss.
test_loss_values = []
for t in range(EPOCHS):
  print(f'Epoch {t+1}\n------------------------')
  train(dataloader, model, loss_function, optimizer)
  test_loss = test(dataloader, model, loss_function)
  test_loss_values.append(test_loss)

# Plot the loss history.
plt.figure()
plt.plot(test_loss_values, '-o', color='#ffbf00')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(axis='y')
plt.show()

# Test the model on new data not part of the dataset.
test_angles = [1, 7, 13, 43, 82]
test_filenames = generate_input_images(test_angles)
for filename, angle in zip(test_filenames, test_angles):
    input_array = torch.tensor(
        np.expand_dims(np.asarray(Image.open(filename), np.uint8), (0, 1))
        )
    os.remove(filename)
    output = model(input_array).detach().numpy()[0, :]
    output = output.reshape(OUTPUT_SIZE, order='F')
    # Convert the output to an RGB array.
    output = np.dstack((output, np.ones(OUTPUT_SIZE, float), 2/3 * np.ones(OUTPUT_SIZE, float)))
    output = hsv_to_rgb(output)
    # Save the generated output image.
    output = output.transpose((1, 0, 2)).astype(np.uint8)
    with Image.fromarray(output) as image:
        image.save(os.path.join(FOLDER_ROOT, f'test_{angle}.png'))