import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.io import read_image


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load data.
batch_size = 4
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(image):
    image = image/2 + 0.5
    image = image.numpy()
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()

images, labels = next(iter(train_dataloader))
imshow(torchvision.utils.make_grid(images))
print(' '.join(classes[i] for i in labels[:batch_size]))

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

network = Network()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch+1}, {i+1}] loss: {running_loss/2000}')
            running_loss = 0.0
print('Done.')

# Save the model.
PATH = './cifar_net.pth'
torch.save(network.state_dict(), PATH)

# Load the model.
network = Network()
network.load_state_dict(torch.load(PATH))

# Test the model.
images, labels = next(iter(test_dataloader))
imshow(torchvision.utils.make_grid(images))
print(' '.join(classes[i] for i in labels[:batch_size]))

outputs = network(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(classes[i] for i in predicted[:4]))

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = network(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )
    
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


# learning_rate = 1e-3
# batch_size = 64
# epochs = 5

# # Load dataset.
# train_data = datasets.FashionMNIST(
#     root='data',  # Folder in which data is stored
#     train=True,  # Get training data instead of testing
#     download=True,  # Download data if not available at root
#     transform=transforms.ToTensor(),
#     target_transform=transforms.Lambda(
#         lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
#         ),
# )
# test_data = datasets.FashionMNIST(
#     root='data',
#     train=False,
#     download=True,
#     transform=transforms.ToTensor(),
# )
# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# def train_loop(dataloader, model, loss_function, optimizer):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         prediction = model(X)
#         loss = loss_function(prediction, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

# def test_loop(dataloader, model, loss_function):
#     size = len(dataloader.dataset)
#     number_batches = len(dataloader)
#     test_loss, correct = 0, 0

#     with torch.no_grad():
#         for X, y in dataloader:
#             prediction = model(X)
#             test_loss += loss_function(prediction, y).item()
#             correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    
#     test_loss /= number_batches
#     correct /= size
#     print(f'Test Error: \n   Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f}\n')

# model = NeuralNetwork().to(device)
# print(model)

# loss_function = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# epochs = 10
# for t in range(epochs):
#     print(f'Epoch {t+1}\n---------------------------')
#     train_loop(train_dataloader, model, loss_function, optimizer)
#     test_loop(test_dataloader, model, loss_function)




# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# prediction_probabilities = nn.Softmax(dim=1)(logits)
# y_prediction = prediction_probabilities.argmax(1)
# print(f'Predicted class: {y_prediction}')

# input_image = torch.rand(3, 28, 28)
# flatten = nn.Flatten()
# flat_image = flatten(input_image)
# layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# hidden1 = nn.ReLU()(hidden1)
# softmax = nn.Softmax(dim=1)
# prediction_probability = softmax(logits)


# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, image_directory, transform=None, target_transform=None):
#         self.image_labels = pd.read_csv(annotations_file)
#         self.image_directory = image_directory
#         self.transform = transform
#         self.target_transform = target_transform
    
#     def __len__(self):
#         return len(self.image_labels)

#     def __getitem__(self, index):
#         image_path = os.path.join(self.image_directory, self.image_labels.iloc[index, 0])
#         image = read_image(image_path)
#         label = self.image_labels.iloc[index, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label


# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }