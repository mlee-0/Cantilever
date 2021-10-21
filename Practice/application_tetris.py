import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')

ROW_COUNT, COLUMN_COUNT = 20, 10


# Dataset that randomly generates 2D matrices and corresponding labels.
class TetrisDataset(Dataset):
  def __init__(self):
    self.SAMPLES = range(ROW_COUNT * COLUMN_COUNT - 1)
    self.WEIGHTS = [(math.floor(i/COLUMN_COUNT)+1)**4 for i in self.SAMPLES]
    self.INDICES_ROWS = np.indices((ROW_COUNT, COLUMN_COUNT))[0]
  
  def __len__(self):
    # Define how many data to use.
    return 100000
  
  # Input argument index not used.
  def __getitem__(self, index):
    # Generate a matrix with a number of blocks filled.
    array = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int)
    filled_count = random.randint(1, ROW_COUNT*COLUMN_COUNT)
    filled_indices = random.choices(self.SAMPLES, weights=self.WEIGHTS, k=filled_count)
    row_indices, column_indices = np.unravel_index(filled_indices, (ROW_COUNT, COLUMN_COUNT), 'C')
    array[row_indices, column_indices] = 1

    # Calculate the labels.
    lines_cleared = np.all(array, axis=1)
    array_copy = array[~lines_cleared, :]
    # Number of cleared lines.
    line_count = np.sum(lines_cleared)
    # Sum of differences between heights of each column.
    heights = array_copy.shape[0] - np.argmax(np.concatenate((array_copy, np.full((1,COLUMN_COUNT), True))), axis=0)
    roughness = sum([abs(heights[i+1] - heights[i]) for i in range(COLUMN_COUNT-1)])
    # Number of holes.
    holes = np.sum(np.logical_not(array_copy[self.INDICES_ROWS[:array_copy.shape[0],:] > (array_copy.shape[0]-heights)]))

    # Normalize outputs.
    line_count /= ROW_COUNT
    roughness /= ROW_COUNT * (COLUMN_COUNT - 1)
    holes /= (ROW_COUNT - 1) * COLUMN_COUNT
    # # Format array and labels to have 3 dimensions for use in CNN.
    # array = np.expand_dims(array, 0)
    labels = torch.Tensor([[line_count, roughness, holes]])  #torch.Tensor([line_count, roughness, holes])
    # # labels = np.expand_dims(labels, 0)

    return array, labels

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.model_1 = nn.Sequential(
        nn.Linear(ROW_COUNT * COLUMN_COUNT, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )
    self.model_2 = nn.Sequential(
        nn.Linear(ROW_COUNT * COLUMN_COUNT, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )
    self.model_3 = nn.Sequential(
        nn.Linear(ROW_COUNT * COLUMN_COUNT, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )
    # self.cnn = nn.Sequential(
    #     nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
    #     nn.BatchNorm2d(4),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
    #     nn.BatchNorm2d(4),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    # )
    # self.linear = nn.Sequential(
    #     nn.Linear(40, (ROW_COUNT - 1) * COLUMN_COUNT + 1)  #nn.Linear(4 * 7 * 7, 1)
    # )
  
  def forward(self, x):
    x = self.flatten(x)
    output_1 = self.model_1(x)
    output_2 = self.model_2(x)
    output_3 = self.model_3(x)
    # x = self.cnn(x)
    # x = x.view(x.size(0), -1)
    # x = self.linear(x)
    # # x *= ROW_COUNT * (COLUMN_COUNT - 1)
    # # x = x.round()
    return output_1, output_2, output_3

learning_rate = 0.1
batch_size = 32

dataset = TetrisDataset()
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = Network().to(device)

loss_function = nn.MSELoss()  #nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Train the model.
epochs = 5

def train(dataloader, model, loss_function, optimizer):
  size = len(dataloader.dataset)
  for batch, (data, label) in enumerate(dataloader):
    data = data.to(device)
    label = label.to(device)
    # Input the data into the model to return an output.
    line_count, roughness, holes = model(data.float())
    # Calculate the loss by comparing to the true value.
    loss_line_count = loss_function(line_count, label[:,:,0].float())  #loss_function(prediction, label.long())
    loss_roughness = loss_function(roughness, label[:,:,1].float())  #loss_function(prediction, label.long())
    loss_holes = loss_function(holes, label[:,:,2].float())  #loss_function(prediction, label.long())
    loss = loss_line_count + loss_roughness + loss_holes

    # Reset gradients of model parameters.
    optimizer.zero_grad()
    # Backpropagate the prediction loss.
    loss.backward()
    # Adjust model parameters.
    optimizer.step()

    if batch % 1000 == 0:
      loss, current = loss.item(), batch * len(data)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_function):
  size = len(dataloader.dataset)
  batch_count = len(dataloader)
  test_loss, accuracy = 0, 0

  with torch.no_grad():
    for data, label in dataloader:
      data = data.to(device)
      label = label.to(device)
      line_count, roughness, holes = model(data.float())
      test_loss += loss_function(line_count, label[:,:,0].float())  #loss_function(prediction, label.long()).item()
      test_loss += loss_function(roughness, label[:,:,1].float())  #loss_function(prediction, label.long()).item()
      test_loss += loss_function(holes, label[:,:,2].float())  #loss_function(prediction, label.long()).item()
      # accuracy += (prediction.argmax(1) == label).type(torch.float).sum().item()  #(prediction.argmax(0) == label).type(torch.float).sum().item()

  test_loss /= batch_count
  accuracy /= size
  print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return accuracy*100, test_loss

accuracy_values, test_loss_values = [], []
for t in range(epochs):
  print(f'Epoch {t+1}\n------------------------')
  train(train_dataloader, model, loss_function, optimizer)
  accuracy, test_loss = test(test_dataloader, model, loss_function)
  accuracy_values.append(accuracy)
  test_loss_values.append(test_loss)

plt.figure()
plt.plot(accuracy_values, '-o', color='#0095ff')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(axis='y')
plt.show()
plt.figure()
plt.plot(test_loss_values, '-o', color='#ffbf00')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(axis='y')
plt.show()


import time
for i in range(1):
  data, label = next(iter(test_dataloader))
  data = data.to(device)
  label = label.to(device)
  t1 = time.time()
  line_count, roughness, holes = model(data.float())
  t2 = time.time()
  print((t2-t1) * 1000)
  # print(prediction.size(), label.size())
  print(line_count.flatten(), label[0].flatten())
  print(roughness.flatten(), label[1].flatten())
  print(holes.flatten(), label[2].flatten())
  # print(label)
  # print(torch.round(prediction * torch.Tensor([ROW_COUNT, ROW_COUNT * (COLUMN_COUNT - 1), (ROW_COUNT - 1) * COLUMN_COUNT])))
  # print(label * torch.Tensor([ROW_COUNT, ROW_COUNT * (COLUMN_COUNT - 1), (ROW_COUNT - 1) * COLUMN_COUNT]))
  # print(prediction.shape)

torch.save(model.state_dict(), 'weights.pth')
