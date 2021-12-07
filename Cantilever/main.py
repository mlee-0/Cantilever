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


import colorsys
from dataclasses import dataclass
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from metrics import *


# A dataclass that stores settings for each parameter.
@dataclass
class Parameter:
    # Name of the parameter.
    name: str
    # Units for the parameter.
    units: str = ''
    # The minimum and maximum values between which samples are generated.
    low: float = 0
    high: float = 0
    # Number of decimal places to which sample values are rounded.
    precision: int = 0

# Settings for each parameter.
length = Parameter(low=2, high=4, precision=3, name='Length', units='m')
height = Parameter(low=1, high=2, precision=3, name='Height', units='m')
elastic_modulus = Parameter(low=1e9, high=200e9, precision=0, name='Elastic Modulus', units='Pa')
load = Parameter(low=10000, high=100000, precision=0, name='Load', units='N')
angle = Parameter(low=0, high=360, precision=2, name='Angle', units='Degrees')
# Names of quantities that are not generated but are still stored in the text files.
key_x_load = 'Load X'
key_y_load = 'Load Y'
key_image_length = 'Image Length'
key_image_height = 'Image Height'

# Size of input images. Must have the same aspect ratio as the largest possible cantilever geometry.
INPUT_CHANNELS = 4
INPUT_SIZE = (50, 25, INPUT_CHANNELS)
assert (INPUT_SIZE[1] / INPUT_SIZE[0]) == (height.high / length.high), 'Input image size must match aspect ratio of cantilever.'
# Size of output images produced by the network. Output images produced by FEA will be resized to this size.
OUTPUT_CHANNELS = 1
OUTPUT_SIZE = (*INPUT_SIZE[:2], OUTPUT_CHANNELS)

# Folders and files.
FOLDER_ROOT = 'Cantilever'
FOLDER_TRAIN_INPUTS = os.path.join(FOLDER_ROOT, 'Train Inputs')
FOLDER_TRAIN_OUTPUTS = os.path.join(FOLDER_ROOT, 'Train Outputs')
FOLDER_TEST_INPUTS = os.path.join(FOLDER_ROOT, 'Test Inputs')
FOLDER_TEST_OUTPUTS = os.path.join(FOLDER_ROOT, 'Test Outputs')
FILENAME_SAMPLES_TRAIN = 'samples_train.txt'
FILENAME_SAMPLES_TEST = 'samples_test.txt'

# Number of digits used for numerical file names.
NUMBER_DIGITS = 6

# Model parameters file name and path.
FILEPATH_MODEL = os.path.join(FOLDER_ROOT, 'model.pth')
# Training hyperparameters.
BATCH_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 100


# Generate sample values for each parameter and return them as a dictionary.
def generate_samples(number_samples, show_histogram=False) -> dict:
    # Generate samples.
    load_samples = np.linspace(load.low, load.high, number_samples)
    angle_samples = np.linspace(angle.low, angle.high, number_samples)
    length_samples = np.linspace(length.low, length.high, number_samples)
    height_samples = np.linspace(height.low, height.high, number_samples)
    elastic_modulus_samples = np.linspace(elastic_modulus.low, elastic_modulus.high, number_samples)
    # Randomize ordering of samples.
    np.random.shuffle(load_samples)
    np.random.shuffle(angle_samples)
    np.random.shuffle(length_samples)
    np.random.shuffle(height_samples)
    np.random.shuffle(elastic_modulus_samples)
    
    # Calculate the image size corresponding to the geometry.
    image_lengths = np.round(INPUT_SIZE[0] * (length_samples / length.high))
    image_heights = np.round(INPUT_SIZE[1] * (height_samples / height.high))
    
    # Calculate the x- and y-components of the loads and corresponding angles.
    x_loads = np.round(
        np.cos(angle_samples * (np.pi/180)) * load_samples / (image_heights-1),
        NUMBER_DIGITS
        )
    y_loads = np.round(
        np.sin(angle_samples * (np.pi/180)) * load_samples / (image_heights-1),
        NUMBER_DIGITS
        )
    
    # Round samples to a fixed number of decimal places.
    load_samples = np.round(load_samples, load.precision)
    angle_samples = np.round(angle_samples, angle.precision)
    length_samples = np.round(length_samples, length.precision)
    height_samples = np.round(height_samples, height.precision)
    elastic_modulus_samples = np.round(elastic_modulus_samples, elastic_modulus.precision)
    x_loads = np.round(x_loads, load.precision)
    y_loads = np.round(y_loads, load.precision)
    
    # Plot histograms for angle samples.
    if show_histogram:
        plt.figure()
        plt.hist(
            angle_samples,
            bins=round(angle.high-angle.low),
            rwidth=0.75,
            color='#0095ff',
            )
        plt.xticks(np.linspace(angle.low, angle.high, 5))
        plt.title(angle.name)
        plt.show()
    
    return {
        load.name: load_samples,
        angle.name: angle_samples,
        length.name: length_samples,
        height.name: height_samples,
        elastic_modulus.name: elastic_modulus_samples,
        key_x_load: x_loads,
        key_y_load: y_loads,
        key_image_length: image_lengths,
        key_image_height: image_heights,
        }

# Write the specified sample values to a text file.
def write_samples(samples, filename) -> None:
    number_samples = get_sample_size(samples)
    text = [None] * number_samples
    for i in range(number_samples):
        text[i] = ','.join(
            [f'{str(i+1).zfill(NUMBER_DIGITS)}'] + [f'{key}:{value[i]}' for key, value in samples.items()]
            ) + '\n'
    with open(os.path.join(FOLDER_ROOT, filename), 'w') as file:
        file.writelines(text)
    print(f'Wrote samples in {filename}.')

# Return the sample values found in the text file previously generated.
def read_samples(filename) -> dict:
    samples = {}
    filename = os.path.join(FOLDER_ROOT, filename)
    try:
        with open(filename, 'r') as file:
            for line in file.readlines():
                for data in line.split(',')[1:]:
                    key, value = data.split(':')
                    key, value = key.strip(), float(value)
                    if key in samples:
                        samples[key].append(value)
                    else:
                        samples[key] = [value]
    except FileNotFoundError:
        print(f'"{filename}" not found.')
        return
    else:
        print(f'Found samples in {filename}.')
        return samples

# Create images for the sample values provided inside the specified folder.
def generate_input_images(samples, folder_inputs) -> None:
    # Remove existing input images in the folder.
    filenames = glob.glob(os.path.join(folder_inputs, '*.png'))
    for filename in filenames:
        os.remove(filename)
    print(f'Deleted {len(filenames)} existing images in {folder_inputs}.')

    for i in range(get_sample_size(samples)):
        pixel_length, pixel_height = int(samples[key_image_length][i]), int(samples[key_image_height][i])
        image = np.zeros((*INPUT_SIZE[1::-1], INPUT_SIZE[2]))
        # Create a channel with a gray line of pixels representing the load magnitude and direction.
        r = np.arange(max(image.shape))
        x = r * np.cos(samples[angle.name][i] * np.pi/180) + image.shape[1]/2
        y = r * np.sin(samples[angle.name][i] * np.pi/180) + image.shape[0]/2
        x = x.astype(int)
        y = y.astype(int)
        inside_image = (x >= 0) * (x < image.shape[1]) * (y >= 0) * (y < image.shape[0])
        image[y[inside_image], x[inside_image], 0] = 255 * (samples[load.name][i] / load.high)
        image[:, :, 0] = np.flipud(image[:, :, 0])
        # Create a channel with a white rectangle representing the dimensions of the cantilever.
        image[:pixel_height, :pixel_length, 1] = 255
        # Create a channel with the elastic modulus distribution.
        image[:pixel_height, :pixel_length, 2] = 255 * (samples[elastic_modulus.name][i] / elastic_modulus.high)
        # Create a channel with the fixed boundary conditions.
        image[:pixel_height, 0, 3] = 255
        # Write image files.
        filename = os.path.join(folder_inputs, f'input_{str(i+1).zfill(NUMBER_DIGITS)}.png')
        with Image.fromarray(image.astype(np.uint8), 'RGBA') as file:
            file.save(filename)
    print(f'Wrote {i+1} input images in {folder_inputs}.')

# Get the number of samples found in the specified sample values.
def get_sample_size(samples) -> int:
    sample_sizes = [len(_) for _ in samples.values()]
    low, high = min(sample_sizes), max(sample_sizes)
    assert low == high, 'Found different numbers of samples in the provided samples:  min. {low}, max. {high}.'
    return low
    
# Write a text file containing ANSYS commands used to automate FEA and generate stress contour images.
def write_ansys_script(samples, filename) -> None:
    number_samples = get_sample_size(samples)
    # Read the template script.
    with open(os.path.join(FOLDER_ROOT, 'ansys_template.lgw'), 'r') as file:
        lines = file.readlines()
    
    # Replace placeholder lines in the template script.
    with open(os.path.join(FOLDER_ROOT, filename), 'w') as file:
        # Initialize dictionary of placeholder strings (keys) and strings they should be replaced with (values).
        placeholder_substitutions = {}
        # Define names of variables.
        loop_variable = 'i'
        samples_variable = 'samples'
        # Add commands that define the array containing generated samples.
        commands_define_samples = [f'*DIM,{samples_variable},ARRAY,{9},{number_samples}\n']
        for i in range(number_samples):
            commands_define_samples.append(
                f'{samples_variable}(1,{i+1}) = {samples[load.name][i]},{samples[key_x_load][i]},{samples[key_y_load][i]},{samples[angle.name][i]},{samples[length.name][i]},{samples[height.name][i]},{samples[elastic_modulus.name][i]},{samples[key_image_length][i]},{samples[key_image_height][i]}\n'
                )
        placeholder_substitutions['! placeholder_define_samples\n'] = commands_define_samples
        # Add loop commands.
        placeholder_substitutions['! placeholder_loop_start\n'] = f'*DO,{loop_variable},1,{number_samples},1\n'
        placeholder_substitutions['! placeholder_loop_end\n'] = f'*ENDDO\n'
        # Add commands that format and create the output files.
        placeholder_substitutions['! placeholder_define_suffix\n'] = f'suffix = \'{"0"*NUMBER_DIGITS}\'\n'
        placeholder_substitutions['! placeholder_define_number\n'] = f'number = CHRVAL({loop_variable})\n'
        placeholder_substitutions['! placeholder_define_filename\n'] = f'filename = \'fea_%STRFILL(suffix,number,{NUMBER_DIGITS}-STRLENG(number)+1)%\'\n'
        # Substitute all commands into placeholders.
        for placeholder in placeholder_substitutions:
            command = placeholder_substitutions[placeholder]
            indices = [i for i, line in enumerate(lines) if line == placeholder]
            if isinstance(command, list):
                for index in indices:
                    for sub_command in command[::-1]:
                        lines.insert(index+1, sub_command)
                    del lines[index]
            else:
                for index in indices:
                    lines[index] = command
        # Write the file.
        file.writelines(lines)
        print(f'Wrote {filename}.')

# Properly order stress data in text files generated by FEA and return them as a 3D array.
def fea_to_array(samples, folder) -> None:
    number_samples = get_sample_size(samples)
    fea_filenames = glob.glob(os.path.join(folder, '*.txt'))
    fea_filenames = sorted(fea_filenames)
    assert len(fea_filenames) == number_samples, f'Found {len(fea_filenames)} .txt files in {folder}, but should be {number_samples}.'
    # Initialize the array to hold all stress values.
    stresses = np.full((*OUTPUT_SIZE[1::-1], OUTPUT_SIZE[2], len(fea_filenames)), -1)
    for i, fea_filename in enumerate(fea_filenames):
        # Get the stress value at each node.
        with open(fea_filename, 'r') as file:
            stress = [float(line) for line in file.readlines()]
        array = np.zeros((int(samples[key_image_height][i]), int(samples[key_image_length][i])))
        # Determine the number of mesh divisions used in this sample.
        mesh_divisions = (int(samples[key_image_length][i]-1), int(samples[key_image_height][i]-1))
        # Interior nodes.
        array[1:-1, 1:-1] = np.flipud(
            np.reshape(stress[2*sum(mesh_divisions):], [_-1 for _ in mesh_divisions[::-1]], 'F')
            )
        # Corner nodes.
        array[-1, 0] = stress[0]
        array[-1, -1] = stress[1]
        array[0, -1] = stress[1+mesh_divisions[0]]
        array[0, 0] = stress[1+mesh_divisions[0]+mesh_divisions[1]]
        # Edge nodes.
        array[-1, 1:-1] = stress[2:2+mesh_divisions[0]-1]
        array[1:-1, -1] = stress[2+mesh_divisions[0]:2+mesh_divisions[0]+mesh_divisions[1]-1][::-1]
        array[0, 1:-1] = stress[2+mesh_divisions[0]+mesh_divisions[1]:2+2*mesh_divisions[0]+mesh_divisions[1]-1][::-1]
        array[1:-1, 0] = stress[2+2*mesh_divisions[0]+mesh_divisions[1]:2+2*mesh_divisions[0]+2*mesh_divisions[1]-1]
        # Insert the array.
        stresses[:array.shape[0], :array.shape[1], :, i] = np.expand_dims(array, array.ndim)
    return stresses

# Convert a 3-channel RGB array into a 1-channel hue array with values in [0, 1].
def rgb_to_hue(array) -> np.ndarray:
    array = array / 255
    hue_array = np.zeros((array.shape[0], array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            hsv = colorsys.rgb_to_hsv(array[i, j, 0], array[i, j, 1], array[i, j, 2])
            hue_array[i, j] = hsv[0]
    return hue_array

# Convert a 3-channel HSV array into a 3-channel RGB array.
def hsv_to_rgb(array) -> np.ndarray:
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            rgb = colorsys.hsv_to_rgb(array[i, j, 0], array[i, j, 1], array[i, j, 2])
            for k in range(3):
                array[i, j, k] = rgb[k] * 255
    return array

# Convert the model's output array to a color image.
def array_to_colormap(array) -> np.ndarray:
    # Invert the values so that red represents high stresses.
    array = 1 - array
    # Constrain the values so that only colors from red to blue are shown, to match standard colors used in FEA.
    array = array * (240/360)
    # Convert the output to an RGB array.
    array = np.dstack((array, np.ones(array.shape, float), 2/3 * np.ones(array.shape, float)))
    array = hsv_to_rgb(array)
    return array

# Dataset that retrieves input and output images for the CNN. Output images are not retrieved if the outputs folder is not specified.
class CantileverDataset(Dataset):
    def __init__(self, folder_inputs, folder_outputs, is_train=False):
        self.folder_inputs = folder_inputs
        self.folder_outputs = folder_outputs
        
        # Get all input images and store each in a list.
        input_filenames = glob.glob(os.path.join(self.folder_inputs, '*.png'))
        input_filenames = sorted(input_filenames)
        self.number_samples = len(input_filenames)
        self.inputs = [None] * self.number_samples
        for i, filename in enumerate(input_filenames):
            array = np.asarray(Image.open(filename), np.uint8) / 255,  # Scale to be <= 1
            array = array[0]
            if array.ndim < 3:
                array = np.expand_dims(array, axis=2)
            self.inputs[i] = np.transpose(array, [2, 0, 1])  # Make channel dimension the first dimension
        
        # Get FEA stress data.
        labels = fea_to_array(
            read_samples(FILENAME_SAMPLES_TRAIN if is_train else FILENAME_SAMPLES_TEST),
            FOLDER_TRAIN_OUTPUTS if is_train else FOLDER_TEST_OUTPUTS,
            )
        # Scale the stress values to be <= 1, but not [0, 1].
        if is_train:
            CantileverDataset.store_stress_range(np.max(labels))
        labels = labels / CantileverDataset.maximum_stress
        # Replace background values.
        labels[labels < 0] = 0
        # Store each stress array in a list.
        self.labels = [None] * labels.shape[-1]
        for i in range(len(self.labels)):
            # # Keep only non-empty regions of array.
            # self.labels[i] = labels[
            #     :int(np.nonzero(np.any(labels[:, :, :, i] >= 0, axis=1))[0][-1] + 1),
            #     :int(np.nonzero(np.any(labels[:, :, :, i] >= 0, axis=0))[0][-1] + 1),
            #     :,
            #     i,
            #     ]
            # Make channel dimension the first dimension.
            self.labels[i] = np.transpose(labels[:, :, :, i], [2, 0, 1])

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
    # Store the maximum stress value found in the training dataset as a class variable to be referenced by the test datset.
    @classmethod
    def store_stress_range(cls, maximum):
        cls.maximum_stress = maximum

# A CNN that predicts the stress contour in a cantilever beam with a point load at the free end.
class StressContourCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=3),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.linear = nn.Sequential(
            nn.Linear(in_features=512, out_features=OUTPUT_SIZE[0]*OUTPUT_SIZE[1]*OUTPUT_CHANNELS),
            )
    
    def forward(self, x):
        x = x.float()
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape((BATCH_SIZE, OUTPUT_CHANNELS, *OUTPUT_SIZE[1::-1]))
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

        # if batch % (BATCH_SIZE * 10) == 0:
        #     loss, current = loss.item(), batch * len(data)
        #     print(f'Loss: {loss:>7f}  (batch {current} of {len(dataloader.dataset)})')

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
    device = 'cpu'  #'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Initialize the model and load its parameters if it has already been trained.
    model = StressContourCnn()
    train_model = True
    if os.path.exists(FILEPATH_MODEL):
        model.load_state_dict(torch.load(FILEPATH_MODEL))
        model.eval()
        print(f'Loaded previously trained parameters from {FILEPATH_MODEL}.')
        if not input('Continue training this model? (y/n) ').lower().startswith('y'):
            train_model = False
    
    # Set up the training data.
    train_dataset = CantileverDataset(FOLDER_TRAIN_INPUTS, FOLDER_TRAIN_OUTPUTS, is_train=True)
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
        
        # Save the model parameters.
        torch.save(model.state_dict(), FILEPATH_MODEL)
        print(f'Saved model parameters to {FILEPATH_MODEL}.')
        
        # Plot the loss history.
        plt.figure()
        plt.plot(test_loss_values, '-o', color='#0095ff')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(axis='y')
        plt.show()

    # Set up the testing data.
    test_dataset = CantileverDataset(FOLDER_TEST_INPUTS, FOLDER_TEST_OUTPUTS, is_train=False)
    test_dataloader = DataLoader(test_dataset, shuffle=True)
    
    # Remove existing output images in the folder.
    filenames = glob.glob(os.path.join(FOLDER_TEST_OUTPUTS, '*.png'))
    for filename in filenames:
        os.remove(filename)
    print(f'Deleted {len(filenames)} existing images in {FOLDER_TEST_OUTPUTS}.')
    
    # Test the model on the input images found in the folder.
    area_metric_values = []
    ks_test_values = []
    max_stress_values = []
    for i, (test_input, label) in enumerate(test_dataloader):
        test_output = model(test_input).detach().numpy()[0, 0, :]
        label = label[0, 0, :].numpy()
        # Write FEA images.
        with Image.fromarray((array_to_colormap(label)).astype(np.uint8)) as image:
            filepath = os.path.join(FOLDER_TEST_OUTPUTS, f'fea_{i+1}.png')
            image.save(filepath)
        # Scale the outputs to the original range of stress values and compare with FEA results.
        area_metric_values.append(area_metric(test_output, label))
        ks_test_values.append(ks_test(test_output, label)[0])
        max_stress_values.append(max_value(test_output, label))
        # Convert the output to a color image.
        test_output = array_to_colormap(test_output)
        # Save the generated output image.
        with Image.fromarray(test_output.astype(np.uint8)) as image:
            image.save(os.path.join(
                FOLDER_TEST_OUTPUTS,
                f'test_{str(i+1).zfill(NUMBER_DIGITS)}.png',
                ))
    print(f'Wrote {len(test_dataloader)} output images and {len(test_dataloader)} corresponding labels in {FOLDER_TEST_OUTPUTS}.')
    # Plot evaluation metrics.
    plt.figure()
    plt.plot(area_metric_values, '*', color='#0095ff')
    plt.xlabel('Sample')
    plt.ylim((0, 1))
    plt.title('Area Metric')
    plt.show()
    plt.figure()
    plt.plot(ks_test_values, '*', color='#0095ff')
    plt.xlabel('Sample')
    plt.ylim((0, 1))
    plt.title('K-S Test')
    plt.show()
    f = plt.figure()
    plt.plot([_[0] for _ in max_stress_values], '*', color='#0095ff', label='CNN')
    plt.plot([_[1] for _ in max_stress_values], 'o', color='#ff4040', label='FEA')
    f.legend()
    plt.xlabel('Sample')
    plt.title('Max. Stress')
    plt.show()