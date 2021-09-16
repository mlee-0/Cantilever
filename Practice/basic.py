import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
prediction_probabilities = nn.Softmax(dim=1)(logits)
y_prediction = prediction_probabilities.argmax(1)
print(f'Predicted class: {y_prediction}')

input_image = torch.rand(3, 28, 28)
flatten = nn.Flatten()
flat_image = flatten(input_image)
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
hidden1 = nn.ReLU()(hidden1)
softmax = nn.Softmax(dim=1)
prediction_


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, image_directory, transform=None, target_transform=None):
        self.image_labels = pd.read_csv(annotations_file)
        self.image_directory = image_directory
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_directory, self.image_labels.iloc[index, 0])
        image = read_image(image_path)
        label = self.image_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Load dataset.
training_data = datasets.FashionMNIST(
    root='data',  # Folder in which data is stored
    train=True,  # Get training data instead of testing
    download=True,  # Download data if not available at root
    transform=transforms.ToTensor(),
    target_transform=transforms.Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
        ),
)
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

