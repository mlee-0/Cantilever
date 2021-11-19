'''
Train a CNN that predicts the stress contour in a cantilever beam with variable dimensions and a point load at the free end with variable magnitude and direction.

The input images represent information about the cantilever and loading using two channels:
- Black background with a white/gray line oriented at the specific angle and a brightness representing the load magnitude. Angle values follow the standard coordinate system and increase counterclockwise. An angle of 0 degrees points directly right, and an angle of 90 degrees points directly up. 
- Black background with a white rectangle representing the length and height of the cantilever.

The output images are stress contours in the cantilever generated by FEA.
- Software: ANSYS (Mechanical APDL)
- Element type: PLANE182 (2D 4-node structural solid)
    - Thickness: 1
- Material: structural steel
    - Elastic modulus: 200 GPa
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


# Dataset size.
NUMBER_SAMPLES = 100

# A dataclass that stores settings for each parameter.
@dataclass
class Parameter:
    # The minimum and maximum values between which samples are generated.
    low: float
    high: float
    # Number of decimal places to which sample values are rounded.
    precision: int
    # Name of the parameter.
    name: str
    # Units for the parameter.
    units: str = ''

# Define the settings for each parameter.
load = Parameter(low=10000, high=100000, precision=0, name='Load', units='N')
angle = Parameter(low=0, high=360, precision=2, name='Angle', units='Degrees')
length = Parameter(low=2, high=4, precision=3, name='Length', units='m')
height = Parameter(low=1, high=2, precision=3, name='Height', units='m')
# The two angle values near which there should be more samples than elsewhere.
ANGLE_PEAKS = (0, 180)

# Size of input images. Must have the same aspect ratio as the largest possible cantilever geometry.
INPUT_SIZE = (100, 50, 2)
assert (INPUT_SIZE[1] / INPUT_SIZE[0]) == (height.high / length.high), 'Input image size must match aspect ratio of cantilever.'
# Size of output images produced by the network. Output images produced by FEA will be resized to this size.
OUTPUT_SIZE = INPUT_SIZE[:2]

# FEA meshing settings.
MESH_DIVISIONS = (99, 49)  # Along (length, height)

# Folders.
FOLDER_ROOT = 'Cantilever'
FOLDER_TRAIN_INPUTS = os.path.join(FOLDER_ROOT, 'Train Inputs')
FOLDER_TRAIN_OUTPUTS = os.path.join(FOLDER_ROOT, 'Train Outputs')
FOLDER_TEST_INPUTS = os.path.join(FOLDER_ROOT, 'Test Inputs')
FOLDER_TEST_OUTPUTS = os.path.join(FOLDER_ROOT, 'Test Outputs')

# Number of digits used for numerical file names.
NUMBER_DIGITS = 6

# Model parameters file name and path.
FILEPATH_MODEL = os.path.join(FOLDER_ROOT, 'model.pth')
# Training hyperparameters.
BATCH_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 100


# Generate sample values for each parameter.
def generate_samples(number_samples, show_histogram=False) -> tuple:
    assert number_samples % 4 == 0, f'Sample size {number_samples} must be divisible by 4 for angles to be generated properly.'
    # Helper function for generating unevenly spaced samples within a defined range. Setting "increasing" to True makes spacings increase as values increase.
    generate_logspace_samples = lambda low, high, increasing: (((
        np.logspace(
            0, 1, round(number_samples/4)+1
            ) - 1) / 9) * (1 if increasing else -1) + (0 if increasing else 1)
        ) * (high - low) + low
    # Generate samples.
    load_samples = np.linspace(load.low, load.high, number_samples)
    angle_samples = np.concatenate((
        generate_logspace_samples(angle.low, (ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2, increasing=True)[:-1],
        generate_logspace_samples((ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2, ANGLE_PEAKS[1], increasing=False)[1:],
        generate_logspace_samples(ANGLE_PEAKS[1], (ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2 + ANGLE_PEAKS[1], increasing=True)[:-1],
        generate_logspace_samples((ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2 + ANGLE_PEAKS[1], angle.high, increasing=False)[1:],
        ))
    length_samples = np.linspace(length.low, length.high, number_samples)
    height_samples = np.linspace(height.low, height.high, number_samples)
    # Randomize ordering of samples.
    np.random.shuffle(load_samples)
    np.random.shuffle(angle_samples)
    np.random.shuffle(length_samples)
    np.random.shuffle(height_samples)
    # Round samples to a fixed number of decimal places.
    load_samples = np.round(load_samples, load.precision)
    angle_samples = np.round(angle_samples, angle.precision)
    length_samples = np.round(length_samples, length.precision)
    height_samples = np.round(height_samples, height.precision)
    
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
    return load_samples, angle_samples, length_samples, height_samples

# Create images for the sample values provided inside the specified folder.
def generate_input_images(samples, folder_inputs):
    # Remove existing input images in the folder.
    filenames = glob.glob(os.path.join(folder_inputs, '*.png'))
    for filename in filenames:
        os.remove(filename)
    print(f'Deleted {len(filenames)} existing images in {folder_inputs}.')

    for i, (load_sample, angle_sample, length_sample, height_sample) in enumerate(zip(*samples)):
        image = np.zeros((INPUT_SIZE[1], INPUT_SIZE[0], INPUT_SIZE[2]))
        # Create a channel with a gray line of pixels representing the load magnitude and direction.
        r = np.arange(max(INPUT_SIZE))
        x = r * np.cos(angle_sample * np.pi/180) + INPUT_SIZE[0]/2
        y = r * np.sin(angle_sample * np.pi/180) + INPUT_SIZE[1]/2
        x = x.astype(int)
        y = y.astype(int)
        inside_image = (x >= 0) * (x < INPUT_SIZE[0]) * (y >= 0) * (y < INPUT_SIZE[1])
        image[y[inside_image], x[inside_image], 0] = 255 * (load_sample / load.high)
        image[:, :, 0] = np.flipud(image[:, :, 0])
        # Create a channel with a white rectangle representing the dimensions of the cantilever.
        image[
            :round(height_sample/height.high * image.shape[0]),
            :round(length_sample/length.high * image.shape[1]),
            1
            ] = 255
        # Write image files.
        filename = os.path.join(folder_inputs, f'input_{str(i+1).zfill(NUMBER_DIGITS)}.png')
        with Image.fromarray(image.astype(np.uint8), 'LA') as file:
            file.save(filename)
    print(f'Wrote {len(samples[0])} input images in {folder_inputs}.')

# Return the x- and y-components of the specified loads and corresponding angles.
def calculate_load_components(load_samples, angle_samples) -> list:
    load_components = []
    for load_sample, angle_sample in zip(load_samples, angle_samples):
        angle_sample *= (np.pi / 180)
        load_components.append((
            np.round(np.cos(angle_sample) * load_sample / (MESH_DIVISIONS[1]), NUMBER_DIGITS),
            np.round(np.sin(angle_sample) * load_sample / (MESH_DIVISIONS[1]), NUMBER_DIGITS),
            ))
    return load_components

# Write a text file containing ANSYS commands used to automate FEA and generate stress contour images.
def write_ansys_script(samples, load_components, filename) -> None:
    number_samples = len(samples[0])
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
        # Add loop commands.
        placeholder_substitutions['! placeholder_loop_start\n'] = f'*DO,{loop_variable},1,{number_samples},1\n'
        placeholder_substitutions['! placeholder_loop_end\n'] = f'*ENDDO\n'
        # Add commands that define the array containing generated samples.
        commands_define_samples = [f'*DIM,{samples_variable},ARRAY,{6},{number_samples}\n']
        for i, (load_sample, angle_sample, length_sample, height_sample, (load_x, load_y)) in enumerate(zip(*samples, load_components)):
            commands_define_samples.append(
                f'{samples_variable}(1,{i+1}) = {load_sample},{load_x},{load_y},{angle_sample},{length_sample},{height_sample}\n'
                # f'{samples_variable}(1,{i+1}) = {samples[0][i]},{load_components[i][0]},{load_components[i][1]},{samples[1][i]},{samples[2][i]},{samples[3][i]}\n'
                )
        placeholder_substitutions['! placeholder_define_samples\n'] = commands_define_samples
        # Add meshing commands.
        placeholder_substitutions['! placeholder_mesh_length\n'] = f'LESIZE,_Y1, , ,{MESH_DIVISIONS[0]}, , , , ,1\n'
        placeholder_substitutions['! placeholder_mesh_height\n'] = f'LESIZE,_Y1, , ,{MESH_DIVISIONS[1]}, , , , ,1\n'
        # Add commands that format and create the output files.
        placeholder_substitutions['! placeholder_define_suffix\n'] = f'suffix = \'{"0"*NUMBER_DIGITS}\'\n'
        placeholder_substitutions['! placeholder_define_number\n'] = f'number = CHRVAL({loop_variable})\n'
        placeholder_substitutions['! placeholder_define_filename\n'] = f'filename = \'stress_%STRFILL(suffix,number,{NUMBER_DIGITS}-STRLENG(number)+1)%\'\n'
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

# Get properly ordered stress data in text files generated by FEA.
def write_fea_spreadsheet(samples, folder, filename):
    number_samples = len(samples[0])
    length_samples, height_samples = samples[2:4]
    fea_filenames = glob.glob(os.path.join(folder, '*.txt'))
    fea_filenames = sorted(fea_filenames)
    assert len(fea_filenames) == number_samples, f'Found {len(fea_filenames)} .txt files in {folder}, but should be {len(length_samples)}.'
    stresses = np.zeros((*OUTPUT_SIZE[::-1], len(fea_filenames)))
    for i, fea_filename in enumerate(fea_filenames):
        with open(fea_filename, 'r') as file:
            stress = [float(line) for line in file.readlines()]
        array = np.zeros(stresses.shape[:2])
        # Interior nodes.
        array[1:-1, 1:-1] = np.flipud(
            np.reshape(stress[2*sum(MESH_DIVISIONS):], [_-1 for _ in MESH_DIVISIONS[::-1]], 'F')
            )
        # Corner nodes.
        array[-1, 0] = stress[0]
        array[-1, -1] = stress[1]
        array[0, -1] = stress[1+MESH_DIVISIONS[0]]
        array[0, 0] = stress[1+MESH_DIVISIONS[0]+MESH_DIVISIONS[1]]
        # Edge nodes.
        array[-1, 1:-1] = stress[2:2+MESH_DIVISIONS[0]-1]
        array[1:-1, -1] = stress[2+MESH_DIVISIONS[0]:2+MESH_DIVISIONS[0]+MESH_DIVISIONS[1]-1][::-1]
        array[0, 1:-1] = stress[2+MESH_DIVISIONS[0]+MESH_DIVISIONS[1]:2+2*MESH_DIVISIONS[0]+MESH_DIVISIONS[1]-1][::-1]
        array[1:-1, 0] = stress[2+2*MESH_DIVISIONS[0]+MESH_DIVISIONS[1]:2+2*MESH_DIVISIONS[0]+2*MESH_DIVISIONS[1]-1]
        # Resize based on the size of the cantilever.
        array_scaled = array - np.min(array)
        array_scaled = array / np.max(array_scaled)
        with Image.fromarray((array_scaled*255).astype(np.uint8), 'L') as image:
            image = image.resize((
                round((length_samples[i] / length.high) * OUTPUT_SIZE[0]),
                round((height_samples[i] / height.high) * OUTPUT_SIZE[1]),
                ))
            array_scaled = np.asarray(image, float) / 255
            array_scaled *= max(stress) - min(stress)
            array_scaled += min(stress)
        # Insert the resized array.
        array[:] = 0
        array[:array_scaled.shape[0], :array_scaled.shape[1]] = array_scaled
        stresses[:, :, i] = array
    # Write values to spreadsheet file.
    filepath = os.path.join(folder, filename)
    stresses.flatten().tofile(
        filepath,
        sep=',',
        )
    print(f'Wrote properly ordered stress data from FEA in {filepath}.')

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

# Convert the model's output array to a color image.
def array_to_colormap(array):
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
        # Get all input filenames.
        self.input_filenames = glob.glob(os.path.join(self.folder_inputs, '*.png'))
        self.input_filenames = sorted(self.input_filenames)
        # Get FEA stress data and scale to [0, 1].
        output_filename = glob.glob(os.path.join(folder_outputs, '*.csv'))[0]
        self.stresses = np.genfromtxt(output_filename, delimiter=',')
        self.stresses = np.reshape(self.stresses, (*OUTPUT_SIZE[::-1], round(self.stresses.size / (OUTPUT_SIZE[0]*OUTPUT_SIZE[1]))))
        # Scale the stress values to [0, 1].
        if is_train:
            self.store_stress_range(np.min(self.stresses), np.max(self.stresses))
            self.stresses -= np.min(self.stresses)
            self.stresses /= np.max(self.stresses)
        else:
            self.stresses -= CantileverDataset.minimum_stress
            self.stresses /= (CantileverDataset.maximum_stress - CantileverDataset.minimum_stress)
        # for i in range(self.stresses.shape[2]):
        #     self.stresses[:, :, i] -= np.min(self.stresses[:, :, i])
        #     self.stresses[:, :, i] /= np.max(self.stresses[:, :, i])

    def __len__(self):
        return len(self.input_filenames)
    
    def __getitem__(self, index):
        input_filename = self.input_filenames[index]
        array_input = np.asarray(Image.open(input_filename), np.uint8)
        array_input = np.transpose(array_input, [2, 0, 1])  # Make channel dimension the first dimension
        array_output = self.stresses[:, :, index].flatten()
        return array_input, array_output
    
    # Store the minimum and maximum stress values found in the training dataset as class variables to be referenced by the test datset.
    @classmethod
    def store_stress_range(cls, minimum, maximum):
        cls.minimum_stress = minimum
        cls.maximum_stress = maximum

# A CNN that predicts the stress contour in a cantilever beam with a point load at the free end.
class StressContourCnn(nn.Module):
    def __init__(self):
        super().__init__()
        # self.se_resnet = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Linear(in_features=46, out_features=46),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=46, out_features=47*2),
        #     nn.Sigmoid(),
        #     )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            # nn.BatchNorm2d(4),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            # nn.BatchNorm2d(4),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            # nn.BatchNorm2d(4),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3, stride=1),
            # nn.BatchNorm2d(2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=160, out_features=OUTPUT_SIZE[0]*OUTPUT_SIZE[1]),
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
        plt.plot(test_loss_values, '-o', color='#ffbf00')
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
    for i, (test_input, label) in enumerate(test_dataloader):
        test_output = model(test_input).detach().numpy()[0, :]
        test_output = test_output.reshape(OUTPUT_SIZE[::-1], order='C')
        label = label.reshape(OUTPUT_SIZE[::-1]).numpy()
        # Write FEA images.
        with Image.fromarray((array_to_colormap(label)).astype(np.uint8)) as image:
            filepath = os.path.join(FOLDER_TEST_OUTPUTS, f'fea_{i+1}.png')
            image.save(filepath)
        # Scale the outputs to the original range of stress values and compare with FEA results.
        test_output_unscaled = test_output.copy()
        area_metric_values.append(area_metric(test_output_unscaled, label))
        ks_test_values.append(ks_test(test_output_unscaled, label)[0])
        # Convert the output to a color image.
        test_output = array_to_colormap(test_output)
        # Save the generated output image.
        with Image.fromarray(test_output.astype(np.uint8)) as image:
            image.save(os.path.join(
                FOLDER_TEST_OUTPUTS,
                f'test_{str(i+1).zfill(NUMBER_DIGITS)}.png',
                ))
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
    print(f'Wrote {len(test_dataloader)} output images in {FOLDER_TEST_OUTPUTS}.')