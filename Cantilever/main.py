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

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from setup import *
from metrics import *


# Model parameters file name and path.
FILEPATH_MODEL = os.path.join(FOLDER_ROOT, 'model.pth')
# Training hyperparameters.
BATCH_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 100


# Dataset that retrieves input and output images for the CNN. Output images are not retrieved if the outputs folder is not specified.
class CantileverDataset(Dataset):
    maximum_stress = None

    def __init__(self, is_train=False):
        # Create input images and store each in a list.
        samples = read_samples(FILENAME_SAMPLES_TRAIN if is_train else FILENAME_SAMPLES_TEST)
        self.number_samples = get_sample_size(samples)
        self.inputs = generate_input_images(samples)
        
        # Create label images from FEA stress data.
        self.labels, maximum_stress = generate_label_images(
            samples,
            FOLDER_TRAIN_OUTPUTS if is_train else FOLDER_TEST_OUTPUTS,
            CantileverDataset.maximum_stress,
            )
        # Store the maximum stress value found in the training dataset as a class variable to be referenced by the test datset.
        if is_train:
            CantileverDataset.maximum_stress = maximum_stress
        # # Write FEA images.
        # for i, label in enumerate(self.labels):
        #     with Image.fromarray((array_to_colormap(label[0, :, :])).astype(np.uint8)) as image:
        #         filepath = os.path.join(FOLDER_TRAIN_OUTPUTS, f'fea_{i+1}.png')
        #         image.save(filepath)

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

# A CNN that predicts the stress contour in a cantilever beam with a point load at the free end.
class StressContourCnn(nn.Module):
    def __init__(self):
        super().__init__()
        # Return a sequence of layers related to convolution.
        convolution = lambda in_channels, out_channels, kernel_size=3, stride=3: nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
        # Return a sequence of layers related to deconvolution.
        deconvolution = lambda in_channels, out_channels, kernel_size=3, stride=3: nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
        
        self.convolution_1 = convolution(in_channels=INPUT_CHANNELS, out_channels=8)
        self.convolution_2 = convolution(in_channels=8, out_channels=16)
        self.deconvolution_1 = deconvolution(in_channels=16, out_channels=8, kernel_size=4)
        self.deconvolution_2 = deconvolution(in_channels=8, out_channels=OUTPUT_CHANNELS, kernel_size=4)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.autoencoder = nn.Sequential(
            nn.Linear(in_features=48, out_features=24),
            nn.Linear(in_features=24, out_features=12),
            nn.Linear(in_features=12, out_features=24),
            nn.Linear(in_features=24, out_features=48),
            )

        self.linear = nn.Sequential(
            nn.Linear(in_features=5050, out_features=np.prod(OUTPUT_SIZE)),
            )
    
    def forward(self, x):
        ### CONV-DECONV
        # x = x.float()
        # # Convolution.
        # x = self.convolution_1(x)
        # x = self.pooling(x)
        # x = self.convolution_2(x)
        # x = self.pooling(x)
        # # Deconvolution.
        # x = self.deconvolution_1(x)
        # x = self.deconvolution_2(x)
        # x = self.deconvolution_3(x)
        # # Fully connected.
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        
        ### AUTOENCODER
        x = x.float()
        # Convolution.
        x = self.convolution_1(x)
        size_1 = x.size()
        x, indices_1 = self.pooling(x)
        x = self.convolution_2(x)
        size_2 = x.size()
        x, indices_2 = self.pooling(x)
        # Autoencoding.
        size_encoding = x.size()
        x = self.autoencoder(x.flatten())
        x = x.reshape(size_encoding)
        # Deconvolution.
        x = self.unpooling(x, indices_2, output_size=size_2)
        x = self.deconvolution_1(x)
        x = self.unpooling(x, indices_1, output_size=size_1)
        x = self.deconvolution_2(x)
        # Fully connected.
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x.reshape((BATCH_SIZE, *OUTPUT_SIZE))

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

# Save model parameters to a file.
def save(model):
    torch.save(model.state_dict(), FILEPATH_MODEL)
    print(f'Saved model parameters to {FILEPATH_MODEL}.')


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
            if t % 10 == 0 and t > 0:
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
    plt.plot([_[1] for _ in max_stress_values], 'o', color='#ff4040', label='FEA')
    plt.plot([_[0] for _ in max_stress_values], '*', color='#0095ff', label='CNN')
    f.legend()
    plt.xlabel('Sample')
    plt.title('Max. Stress')
    plt.show()