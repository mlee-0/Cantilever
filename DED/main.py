'''
Train and test the model.
'''


import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import metrics
from networks import *
from setup import *


# Model parameters file path.
FILEPATH_MODEL = os.path.join(FOLDER_ROOT, 'model.pth')


class DedDataset(Dataset):
    """Dataset that gets input and label images during training."""
    def __init__(self, dataset: str):
        if dataset == "train":
            sample_indices = [
                _ for _ in range(TOTAL_SAMPLES)
                if _ not in VALIDATE_SAMPLE_INDICES and _ not in TEST_SAMPLE_INDICES
            ]
        elif dataset == "validate":
            sample_indices = VALIDATE_SAMPLE_INDICES
        elif dataset == "test":
            sample_indices = TEST_SAMPLE_INDICES
        else:
            print(f"Invalid dataset name: {dataset}")
            
        self.inputs = read_inputs(FILENAME_DATASET, SHEET_INDEX, sample_indices=sample_indices)
        self.labels = read_labels(FOLDER_LABELS, sample_indices=sample_indices)
        
        # Perform data augmentation.
        augmented_labels = []
        for label in self.labels:
            flipped_label = label.copy()
            flipped_label[0, ...] = np.fliplr(flipped_label[0, ...])
            augmented_labels.append(flipped_label)
        # Insert the augmented labels and corresponding input images.
        index = 1
        for input_image, augmented_label in zip(self.inputs, augmented_labels):
            # Insert after its corresponding image.
            self.inputs.insert(index, input_image)
            self.labels.insert(index, augmented_label)
            index += 2
        
        assert len(self.inputs) == len(self.labels), f"Number of inputs {len(self.inputs)} does not match number of labels {len(self.labels)}"
        self.number_samples = len(self.inputs)

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        # Return copies of arrays so that arrays are not modified.
        return np.copy(self.inputs[index]), np.copy(self.labels[index])

def save(epoch, model, optimizer, loss_history) -> None:
    """Save model parameters to a file."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history,
    }, FILEPATH_MODEL)
    print(f'Saved model parameters to {FILEPATH_MODEL}.')

def main(epoch_count: int, learning_rate: float, batch_size: int, Model: nn.Module, training_split: float, keep_training=None, test_only=False, queue=None, queue_to_main=None):
    """
    Train and test the model.

    `desired_subset_size`: Number of samples to include in the subset. Enter 0 to use all samples found instead of creating a subset.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Initialize the model and optimizer and load their parameters if they have been saved previously.
    model = Model()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    epochs = range(1, epoch_count+1)
    previous_test_loss = []
    if os.path.exists(FILEPATH_MODEL):
        if not test_only:
            if keep_training is None:
                keep_training = input(f'Continue training the model in {FILEPATH_MODEL}? [y/n] ') == 'y'
        else:
            keep_training = True
        
        if keep_training:
            checkpoint = torch.load(FILEPATH_MODEL, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1
            epochs = range(epoch, epoch+epoch_count)
            previous_test_loss = checkpoint['loss']
    else:
        keep_training = False
        test_only = False
    
    train_dataset = DedDataset(dataset="train")
    validate_dataset = DedDataset(dataset="validate")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    size_train_dataset = len(train_dataloader)
    size_validate_dataset = len(validate_dataloader)

    if not test_only:
        if queue:
            queue.put([(epochs[0]-1, epochs[-1]), None, None, None, None])

        validation_loss = []
        for epoch in epochs:
            print(f'Epoch {epoch}\n------------------------')
            
            # Train on the training dataset.
            model.train(True)
            for batch, (data, label) in enumerate(train_dataloader, 1):
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

                if (batch) % 10 == 0:
                    print(f"Training batch {batch}/{size_train_dataset} with loss {loss:,.0f}...", end="\r")
                    if queue:
                        queue.put([None, (batch, size_train_dataset+size_validate_dataset), None, None, None])
            print()

            # Train on the validation dataset. Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
            model.train(False)
            loss = 0
            with torch.no_grad():
                for batch, (data, label) in enumerate(validate_dataloader, 1):
                    data = data.to(device)
                    label = label.to(device)
                    output = model(data)
                    loss += loss_function(output, label.float())
                    if (batch) % 100 == 0:
                        print(f"Validating batch {batch}/{size_validate_dataset}...", end="\r")
                        if queue:
                            queue.put([None, (size_train_dataset+batch, size_train_dataset+size_validate_dataset), None, None, None])
            print()
            loss /= size_validate_dataset
            validation_loss.append(loss)
            print(f"Average loss: {loss:,.0f}")

            # Save the model parameters periodically.
            if (epoch) % 1 == 0:
                save(epoch, model, optimizer, [*previous_test_loss, *validation_loss])
            
            if queue:
                queue.put([(epoch, epochs[-1]), None, epochs, validation_loss, previous_test_loss])
            
            if queue_to_main:
                if not queue_to_main.empty():
                    # Stop training.
                    if queue_to_main.get() == True:
                        queue_to_main.queue.clear()
                        break
        
        # Save the model parameters.
        save(epoch, model, optimizer, [*previous_test_loss, *validation_loss])
        
        # Plot the loss history.
        if not queue:
            plt.figure()
            if previous_test_loss:
                plt.plot(range(1, epochs[0]), previous_test_loss, 'o', color=Colors.GRAY_LIGHT)
            plt.plot(epochs, validation_loss, '-o', color=Colors.BLUE)
            plt.ylim(bottom=0)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(axis='y')
            plt.show()

    # Set up the testing data.
    test_dataset = DedDataset(dataset="test")
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    size_test_dataset = len(test_dataloader)
    
    # Test on the testing dataset.
    model.train(False)
    test_labels = []
    test_outputs = []
    with torch.no_grad():
        for batch, (test_input, label) in enumerate(test_dataloader, 1):
            test_input = test_input.to(device)
            label = label.to(device)
            test_output = model(test_input)
            test_output = test_output[0, :, ...].cpu().detach().numpy()
            label = label[0, :, ...].cpu().numpy()
            test_labels.append(label)
            test_outputs.append(test_output)
            
            # Write the combined label and model output image.
            image = np.vstack((label.transpose((1, 2, 0)), test_output.transpose((1, 2, 0))))
            if OUTPUT_CHANNELS == 1:
                image = np.dstack((image,)*3)
            write_image(image,
                os.path.join(FOLDER_ROOT, f'{batch:03d}.png'),
                )
            
            if queue:
                queue.put([None, (batch, size_test_dataset), None, None, None])
    
    print(f"Wrote {size_test_dataset} test images in {FOLDER_ROOT}.")

    # # Calculate and plot evaluation metrics.
    # if not queue:
    #     # plt.rc('font', family='Source Code Pro', size=10.0, weight='semibold')

    #     # Area metric.
    #     plt.figure()
    #     NUMBER_COLUMNS = 4
    #     for i, (test_output, test_label) in enumerate(zip(test_outputs, test_labels)):
    #         plt.subplot(math.ceil(len(test_outputs) / NUMBER_COLUMNS), NUMBER_COLUMNS, i+1)
    #         cdf_network, cdf_label, bin_edges, area_difference = metrics.area_metric(test_output, test_label, max_values[channel])
    #         plt.plot(bin_edges[1:], cdf_network, '-', color=Colors.BLUE)
    #         plt.plot(bin_edges[1:], cdf_label, ':', color=Colors.RED)
    #         if i == 0:
    #             plt.legend(["CNN", "FEA"])
    #         plt.grid(visible=True, axis='y')
    #         plt.xticks([0, max_values[channel]])
    #         plt.yticks([0, 1])
    #         plt.title(f"[#{i+1}] {area_difference:0.2f}", fontsize=10, fontweight='bold')
    #     plt.suptitle(f"Area Metric", fontweight='bold')
    #     plt.tight_layout()  # Increase spacing between subplots
    #     plt.show()

    #     # Single-value error metrics.
    #     mv, me, mae, mse, mre = [], [], [], [], []
    #     results = {"Maximum Value": mv, "Mean Error": me, "Mean Absolute Error": mae, "Mean Squared Error": mse, "Mean Relative Error": mre}
    #     for test_output, test_label in zip(test_outputs, test_labels):
    #         test_output, test_label = test_output, test_label
    #         mv.append(metrics.maximum_value(test_output, test_label))
    #         me.append(metrics.mean_error(test_output, test_label))
    #         mae.append(metrics.mean_absolute_error(test_output, test_label))
    #         mse.append(metrics.mean_squared_error(test_output, test_label))
    #         mre.append(metrics.mean_relative_error(test_output, test_label))
        
    #     sample_numbers = range(1, len(test_outputs)+1)
    #     plt.figure()
    #     for i, (metric, result) in enumerate(results.items()):
    #         plt.subplot(3, 2, i+1)
    #         plt.grid()
    #         if isinstance(result[0], tuple):
    #             plt.plot(sample_numbers, [_[1] for _ in result], 'o', color=Colors.RED, label="FEA")
    #             plt.plot(sample_numbers, [_[0] for _ in result], '.', color=Colors.BLUE, label="CNN")
    #         else:
    #             plt.plot(sample_numbers, result, '.', markeredgewidth=5, color=Colors.BLUE)
    #             average = np.mean(result)
    #             plt.axhline(average, color=Colors.BLUE_LIGHT, label=f"{average:.2f} average")
    #         plt.legend()
    #         plt.xlabel("Sample Number")
    #         plt.xticks(sample_numbers)
    #         plt.title(metric)
    #     plt.show()


if __name__ == '__main__':
    # Training hyperparameters.
    EPOCHS = 50
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 1
    Model = Nie

    TRAINING_SPLIT = 0.8

    main(EPOCHS, LEARNING_RATE, BATCH_SIZE, Model, TRAINING_SPLIT, keep_training=False, test_only=not True)