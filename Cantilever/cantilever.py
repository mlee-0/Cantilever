'''
Train a CNN that predicts the stress contour in a fixed-size cantilever beam with a point load at the end that varies in magnitude and direction.

Properties of the cantilever beam:
    - Dimensions: 1 m (length), 0.25 m (height), 0.25 m (depth).
    - Material: structural steel.
    - Elastic modulus: 200 GPa.

The input images are black-and-white binary images with a single white pixel representing the load magnitude and load angle. The vertical position of the white pixel represents the magnitude (N), and the horizontal position represents the angle (degrees). Angle values increase clockwise, and an angle of 0 degrees points directly right, while an angle of 90 degrees points directly up. The output images are stress contours produced by the load. 
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
# import torchvision


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')


# Dataset size.
NUMBER_SAMPLES = 20
# Minimum and maximum values of parameters to be varied.
LOAD_RANGE = (1e4, 1e5)
ANGLE_RANGE = (0, 360)
# Increments used to generate samples within the above ranges.
LOAD_STEP = 500
ANGLE_STEP = 5
# Number of possible points within the ranges using the specified increments. The maximum value is excluded from the range.
INPUT_SIZE = (
    round((LOAD_RANGE[1] - LOAD_RANGE[0]) / LOAD_STEP),
    round((ANGLE_RANGE[1] - ANGLE_RANGE[0]) / ANGLE_STEP),
    )
# Size to which output images are resized to account for different output image sizes.
OUTPUT_SIZE = (100, 25)

# Folders.
FOLDER_ROOT = 'Cantilever'
FOLDER_INPUTS = os.path.join(FOLDER_ROOT, 'Inputs')
FOLDER_OUTPUTS = os.path.join(FOLDER_ROOT, 'Outputs')
# Files.
FILENAME_TEXT = 'cantilever_samples.txt'

# Training hyperparameters.
BATCH_SIZE = 5
LEARNING_RATE = 0.1
EPOCHS = 30


# Return randomly generated sample values for load magnitudes and angles and write them to a text file.
def generate_samples():
    # Generate equally spaced values within the corresponding range, using the specified increments.
    load_samples = np.arange(LOAD_RANGE[0], LOAD_RANGE[1], LOAD_STEP)
    angle_samples = np.arange(ANGLE_RANGE[0], ANGLE_RANGE[1], ANGLE_STEP)
    # Randomize the ordering of the samples.
    random.shuffle(load_samples)
    random.shuffle(angle_samples)
    # Keep only the first N samples.
    load_samples, angle_samples = load_samples[:NUMBER_SAMPLES], angle_samples[:NUMBER_SAMPLES]
    # Write samples to text file.
    text = [f'Load: {load},  Angle: {angle}\n' for load, angle in zip(load_samples, angle_samples)]
    with open(os.path.join(FOLDER_ROOT, FILENAME_TEXT), 'w') as file:
        file.writelines(text)
    return load_samples, angle_samples

# Return the sample values found in the text file previously generated.
def read_samples():
    load_samples = []
    angle_samples = []
    filename = os.path.join(FOLDER_ROOT, FILENAME_TEXT)
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            for string in file.readlines():
                load, angle = [int(float(string.split(':')[1])) for string in string.split(',')]
                load_samples.append(load)
                angle_samples.append(angle)
    return load_samples, angle_samples

# Create images for each pair of sample values provided.
def generate_input_images(load_samples, angle_samples):
    filenames = []
    for load, angle in zip(load_samples, angle_samples):
        load, angle = int(load), int(angle)
        # Create a black image and add one white pixel.
        image = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1]))
        image[
            int(round((load - LOAD_RANGE[0]) / LOAD_STEP)),
            int(round((angle - ANGLE_RANGE[0]) / ANGLE_STEP))
            ] = 255
        # Write image files.
        filename = os.path.join(FOLDER_INPUTS, f'cantilever_{load}_{angle}.png')
        with Image.fromarray(image.astype(np.uint8), 'L').convert('1') as file:
            file.save(filename)
            filenames.append(filename)
    return filenames

# Crop the stress contour region of the output images.
def crop_output_images():
    filenames = glob.glob(os.path.join(FOLDER_OUTPUTS, '*.png'))
    for filename in filenames:
        with Image.open(filename) as image:
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
    # array = array / 255
    # max_channels = [np.max(array[:, :, i]) for i in range(3)]
    # min_channels = [np.min(array[:, :, i]) for i in range(3)]
    # max_all = max(max_channels)
    # min_all = min(min_channels)
    # if max_all == max_channels[0]:
    #     hue_array = (array[:, :, 1] - array[:, :, 2]) / (max_all - min_all)
    # elif max_all == max_channels[1]:
    #     hue_array = 2.0 + (array[:, :, 2] - array[:, :, 0]) / (max_all - min_all)
    # elif max_all == max_channels[2]:
    #     hue_array = 4.0 + (array[:, :, 0] - array[:, :, 1]) / (max_all - min_all)
    # else:
    #     raise Exception('Hues could not be calculated.')
    # hue_array *= 60
    # if np.any(hue_array < 0):
    #     hue_array += 360
    # hue_array /= 360
    return hue_array

# Try to read sample values from the text file if it already exists. If not, generate the samples.
load_samples, angle_samples = read_samples()
if not load_samples or not angle_samples:
    load_samples, angle_samples = generate_samples()
# generate_input_images(load_samples, angle_samples)
# Determine the x and y components of the load for easier entry in ANSYS.
for load, angle in zip(load_samples, angle_samples):
    angle *= (np.pi / 180)
    print(f'Load {int(load)} has components: x = {np.cos(angle) * load :.2f}, y = {np.sin(angle) * load :.2f}')
# Crop output images.
crop_output_images()

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
            nn.Linear(in_features=2752, out_features=OUTPUT_SIZE[0]*OUTPUT_SIZE[1]),
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
            print(f'Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

def test(dataloader, model, loss_function):
    batch_count = len(dataloader)
    test_loss, accuracy = 0, 0

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            test_loss += loss_function(output, label.float())

    test_loss /= batch_count
    accuracy /= size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy*100, test_loss

# Set up the data and model.
dataset = CantileverDataset(FOLDER_INPUTS, FOLDER_OUTPUTS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
size = len(dataloader.dataset)
model = StressContourNetwork()
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Train the model and record the accuracy and loss.
accuracy_values, test_loss_values = [], []
for t in range(EPOCHS):
  print(f'Epoch {t+1}\n------------------------')
  train(dataloader, model, loss_function, optimizer)
  accuracy, test_loss = test(dataloader, model, loss_function)
  accuracy_values.append(accuracy)
  test_loss_values.append(test_loss)

# Plot the loss history.
plt.figure()
plt.plot(test_loss_values, '-o', color='#ffbf00')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(axis='y')
plt.show()

# Test the model on new data not part of the dataset.
test_filenames = generate_input_images([50000]*4, [45, 135, 225, 315])
for filename in test_filenames:
    input = np.expand_dims(np.asarray(Image.open(filename), np.uint8), (0, 1))
    os.remove(filename)
    input = torch.tensor(input)
    output = model(input).detach().numpy()[0, :]
    output = output.reshape(OUTPUT_SIZE, order='F')
    # Convert the output to an RGB array.
    output = np.dstack((output, np.ones(OUTPUT_SIZE, float), 2/3 * np.ones(OUTPUT_SIZE, float)))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # print(output[i, j, 0], output[i, j, 1], output[i, j, 2])
            rgb = colorsys.hsv_to_rgb(output[i, j, 0], output[i, j, 1], output[i, j, 2])
            for k in range(3):
                output[i, j, k] = rgb[k] * 255
    # Display the generated output image.
    output = output.transpose((1, 0, 2)).astype(np.uint8)
    plt.figure()
    plt.imshow(output)
    plt.show()