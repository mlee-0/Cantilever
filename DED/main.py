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

# Values used to represent real and fake images for the GAN.
LABEL_REAL = 1
LABEL_FAKE = 0


class DedDataset(Dataset):
    """Dataset that gets input and label images during training."""
    def __init__(self, dataset: str):
        print(f"Creating {dataset} dataset...")

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

def save(filepath: str, epoch: int, models: list, optimizers: list, loss_histories: list) -> None:
    """Save parameters to the given file."""
    torch.save({
        "epoch": epoch,
        **{f"model_{i}": model.state_dict() for i, model in enumerate(models)},
        **{f"optimizer_{i}": optimizer.state_dict() for i, optimizer in enumerate(optimizers)},
        **{f"loss_{i}": loss for i, loss in enumerate(loss_histories)},
    }, filepath)
    print(f"Saved model parameters to {filepath}.")

def load(filepath: str, device: str, models: list, optimizers: list, loss_histories: list) -> tuple:
    """Load the parameters stored in the given file. Return the epoch number, but load the parameters in-place."""
    checkpoint = torch.load(filepath, map_location=torch.device(device))

    for model, key in zip(models, sorted([key for key in checkpoint if key.startswith("model")])):
        model.load_state_dict(checkpoint[key])
    for optimizer, key in zip(optimizers, sorted([key for key in checkpoint if key.startswith("optimizer")])):
        optimizer.load_state_dict(checkpoint[key])
    for loss, key in zip(loss_histories, sorted([key for key in checkpoint if key.startswith("loss")])):
        loss.extend(checkpoint[key])
    
    epoch = checkpoint["epoch"] + 1
    
    return epoch

def create_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create and return the DataLoader objects for the training, validation, and testing datasets."""

    train_dataset = DedDataset(dataset="train")
    validate_dataset = DedDataset(dataset="validate")
    test_dataset = DedDataset(dataset="test")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    
    return train_dataloader, validate_dataloader, test_dataloader

def initialize_weights(model: nn.Module):
    """Custom weight initialization."""
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def train_gan(device: str, model_cnn: nn.Module, model_generator: nn.Module, model_discriminator: nn.Module, learning_rate: float, epoch_count: int, train_dataloader: DataLoader, validate_dataloader: DataLoader, train_existing: bool, test_only: bool, queue = None, queue_from_gui = None) -> None:
    """Train a network consisting of a CNN and a GAN on the given training and validation datasets."""

    size_train_dataset = len(train_dataloader)
    size_validate_dataset = len(validate_dataloader)

    # Initialize the optimizers and loss function.
    optimizer_cnn = torch.optim.SGD(params=model_cnn.parameters(), lr=learning_rate)
    optimizer_generator = torch.optim.Adam(params=model_generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(params=model_discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    loss_function_cnn = nn.MSELoss()
    loss_function_gan = nn.BCELoss()

    # Load their parameters if they have been saved previously.
    epoch = 1
    previous_validation_losses_cnn = []
    previous_validation_losses_generator = []
    previous_validation_losses_discriminator = []
    if os.path.exists(FILEPATH_MODEL):
        if not test_only:
            if train_existing is None:
                train_existing = input(f'Continue training the model in {FILEPATH_MODEL}? [y/n] ') == 'y'
        else:
            train_existing = True
        
        if train_existing:
            epoch = load(
                FILEPATH_MODEL,
                device,
                (model_cnn, model_generator, model_discriminator),
                (optimizer_generator, optimizer_discriminator),
                (previous_validation_losses_cnn, previous_validation_losses_generator, previous_validation_losses_discriminator),
            )
    else:
        train_existing = False
        test_only = False
    
    epochs = range(epoch, epoch+epoch_count)
    if queue:
        queue.put([(epochs[0]-1, epochs[-1]), None, None, None, None])
    
    # Initialize the validation losses.
    validation_losses_cnn = []
    validation_losses_generator = []
    validation_losses_discriminator = []

    # Main training-validation loop.
    for epoch in epochs:
        print(f'Epoch {epoch}\n------------------------')

        model_cnn.train(True)
        model_generator.train(True)
        model_discriminator.train(True)

        # Train on the training dataset.
        for batch, (data, label) in enumerate(train_dataloader, 1):
            # Add random noise to label images for the first # epochs to improve stability of GAN training.
            EPOCHS_RANDOM_NOISE = 50
            if epoch <= EPOCHS_RANDOM_NOISE:
                std = 0.1 * max([1 - epoch/EPOCHS_RANDOM_NOISE, 0])  # Decays linearly from 1.0 to 0.0
                mean = 0.0
                label = label + torch.randn(label.size(), device=device) * std + mean
            
            train_gan_epoch(data, label, device, model_cnn, model_generator, model_discriminator, optimizer_cnn, optimizer_generator, optimizer_discriminator, loss_function_cnn, loss_function_gan)

            # Periodically display progress.
            if (batch) % 5 == 0:
                print(f"Training batch {batch}/{size_train_dataset}...", end="\r")
                if queue:
                    queue.put([None, (batch, size_train_dataset+size_validate_dataset), None, None, None])
        
        model_cnn.train(False)
        model_generator.train(False)
        model_discriminator.train(False)
        
        # Test on the validation dataset.
        cumulative_loss_cnn = 0
        cumulative_loss_generator = 0
        cumulative_loss_discriminator = 0
        with torch.no_grad():
            for batch, (data, label) in enumerate(validate_dataloader, 1):
                loss_cnn, loss_generator, loss_discriminator = test_gan_epoch(data, label, device, model_cnn, model_generator, model_discriminator, loss_function_cnn, loss_function_gan)

                cumulative_loss_cnn += loss_cnn
                cumulative_loss_generator += loss_generator
                cumulative_loss_discriminator += loss_discriminator

                if (batch) % 5 == 0:
                    print(f"Validating batch {batch}/{size_validate_dataset}...", end="\r")
                    if queue:
                        queue.put([None, (size_train_dataset+batch, size_train_dataset+size_validate_dataset), None, None, None])
        print()

        loss_cnn = cumulative_loss_cnn / size_validate_dataset
        loss_generator = cumulative_loss_generator / size_validate_dataset
        loss_discriminator = cumulative_loss_discriminator / size_validate_dataset
        validation_losses_cnn.append(loss_cnn)
        validation_losses_generator.append(loss_generator)
        validation_losses_discriminator.append(loss_discriminator)
        print(f"Average loss: {loss_cnn:,.2f} (CNN), {loss_generator:,.2f} (generator), {loss_discriminator:,.2f} (discriminator)")

        # Save the model parameters periodically.
        if (epoch) % 1 == 0:
            save(
                FILEPATH_MODEL,
                epoch,
                [model_cnn, model_generator, model_discriminator],
                [optimizer_generator, optimizer_discriminator],
                [
                    [*previous_validation_losses_generator, *validation_losses_generator],
                    [*previous_validation_losses_discriminator, *validation_losses_discriminator],
                    [*previous_validation_losses_cnn, *validation_losses_cnn],
                ],
            )
        
        # if queue:
        #     queue.put([(epoch, epochs[-1]), None, epochs, validation_losses, previous_validation_losses])
        
        # if queue_from_gui:
        #     if not queue_from_gui.empty():
        #         # Stop training.
        #         if queue_from_gui.get() == True:
        #             queue_from_gui.queue.clear()
        #             break
    
    # Save the model parameters.
    save(
        FILEPATH_MODEL,
        epoch,
        [model_cnn, model_generator, model_discriminator],
        [optimizer_generator, optimizer_discriminator],
        [[*previous_validation_losses_generator, *validation_losses_generator], [*previous_validation_losses_discriminator, *validation_losses_discriminator]],
    )
    
    # Plot the loss history.
    if not queue:
        plt.figure()
        if previous_validation_losses_cnn and previous_validation_losses_generator and previous_validation_losses_discriminator:
            # plt.plot(range(1, epochs[0]), previous_validation_losses_cnn, 'x', color=Colors.GRAY_LIGHT)
            plt.plot(range(1, epochs[0]), previous_validation_losses_generator, 'o', color=Colors.GRAY_LIGHT)
            plt.plot(range(1, epochs[0]), previous_validation_losses_discriminator, '*', color=Colors.GRAY_LIGHT)
        # plt.plot(epochs, validation_losses_cnn, '-x', color=Colors.BLUE)
        plt.plot(epochs, validation_losses_generator, '-o', color=Colors.BLUE, label="Generator")
        plt.plot(epochs, validation_losses_discriminator, '-*', color=Colors.ORANGE, label="Discriminator")
        plt.legend()
        plt.ylim(bottom=0)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(axis='y')
        plt.show()

def train_gan_epoch(data, label, device, model_cnn, model_generator, model_discriminator, optimizer_cnn, optimizer_generator, optimizer_discriminator, loss_function_cnn, loss_function_gan):
    """Train the GAN for one epoch and return the loss values."""

    data = data.to(device)
    label = label.to(device)
    label = label.float()

    # Reset gradients.
    optimizer_cnn.zero_grad()
    optimizer_generator.zero_grad()
    optimizer_discriminator.zero_grad()

    # Train the discriminator with an all-real batch.
    model_discriminator.zero_grad()
    # Create a tensor of labels for each image in the batch.
    label_discriminator = torch.full((label.size(0),), LABEL_REAL, dtype=torch.float, device=device)
    # Forward pass real images through the discriminator.
    output_discriminator_real = model_discriminator(label).view(-1)
    # Calculate the loss of the discriminator.
    loss_real = loss_function_gan(output_discriminator_real, label_discriminator)
    # Calculate gradients.
    loss_real.backward()

    # Train the discriminator with an all-fake batch.
    # output_cnn = model_cnn(data)
    output_cnn = torch.randn(label.size(0), 100, 1, 1, device=device)
    output_generator = model_generator(output_cnn)
    # Forward pass fake images through the discriminator.
    output_discriminator_fake = model_discriminator(output_generator.detach()).view(-1)
    # Calculate the loss of the discriminator.
    label_discriminator[:] = LABEL_FAKE
    loss_fake = loss_function_gan(output_discriminator_fake, label_discriminator)
    # Calculate gradients.
    loss_fake.backward()

    # Calculate total loss of discriminator by summing real and fake losses.
    loss_discriminator = loss_real + loss_fake
    # Update discriminator.
    optimizer_discriminator.step()

    # Train the generator.
    model_generator.zero_grad()
    # Forward pass fake images through the discriminator.
    output_discriminator = model_discriminator(output_generator).view(-1)
    # Calculate the loss of the generator, assuming images are real in order to calculate loss correctly.
    label_discriminator[:] = LABEL_REAL
    loss_generator = loss_function_gan(output_discriminator, label_discriminator)
    # Calculate gradients.
    loss_generator.backward()
    # Update generator.
    optimizer_generator.step()

    # Train the CNN.
    loss_cnn = loss_function_cnn(output_generator, label)
    # loss_cnn.backward()
    # optimizer_cnn.step()

    return loss_cnn, loss_generator, loss_discriminator

def test_gan_epoch(data, label, device, model_cnn, model_generator, model_discriminator, loss_function_cnn, loss_function_gan) -> tuple:
    """Test the GAN for one epoch and return the loss values."""

    data = data.to(device)
    label = label.to(device)
    label = label.float()

    output_cnn = torch.randn(label.size(0), 100, 1, 1, device=device)  # model_cnn(data)
    output_generator = model_generator(output_cnn)
    
    output_discriminator_real = model_discriminator(label).view(-1)
    output_discriminator_fake = model_discriminator(output_generator).view(-1)
    label_discriminator_real = torch.full((label.size(0),), LABEL_REAL, dtype=torch.float, device=device)
    label_discriminator_fake = torch.full((label.size(0),), LABEL_FAKE, dtype=torch.float, device=device)

    loss_cnn = loss_function_cnn(output_generator, label)
    loss_generator = loss_function_gan(output_discriminator_fake, label_discriminator_real)
    loss_discriminator = loss_function_gan(output_discriminator_real, label_discriminator_real) + loss_function_gan(output_discriminator_fake, label_discriminator_fake)

    return loss_cnn, loss_generator, loss_discriminator

# def train(device, model, learning_rate, epoch_count, train_dataloader, validate_dataloader, train_existing: bool, test_only: bool, queue=None, queue_from_gui=None) -> None:
#     """Train a single model on the given training and validation datasets."""

#     size_train_dataset = len(train_dataloader)
#     size_validate_dataset = len(validate_dataloader)

#     # Initialize the optimizer and loss function.
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     loss_function = nn.MSELoss()

#     # Load their parameters if they have been saved previously.
#     epoch = 1
#     previous_validation_losses = []
#     if os.path.exists(FILEPATH_MODEL):
#         if not test_only:
#             if train_existing is None:
#                 train_existing = input(f'Continue training the model in {FILEPATH_MODEL}? [y/n] ') == 'y'
#         else:
#             train_existing = True
        
#         if train_existing:
#             epoch = load(FILEPATH_MODEL, device, [model], [optimizer], [previous_validation_losses])
#     else:
#         train_existing = False
#         test_only = False
    
#     epochs = range(epoch, epoch+epoch_count)
#     if queue:
#         queue.put([(epochs[0]-1, epochs[-1]), None, None, None, None])
    
#     validation_losses = []
#     for epoch in epochs:
#         print(f'Epoch {epoch}\n------------------------')
        
#         model.train(True)

#         # Train on the training dataset.
#         for batch, (data, label) in enumerate(train_dataloader, 1):
#             data = data.to(device)
#             label = label.to(device)
#             output = model(data)
#             loss = loss_function(output, label.float())
#             # Reset gradients of model parameters.
#             optimizer.zero_grad()
#             # Backpropagate the prediction loss.
#             loss.backward()
#             # Adjust model parameters.
#             optimizer.step()

#             if (batch) % 10 == 0:
#                 print(f"Training batch {batch}/{size_train_dataset} with loss {loss:,.0f}...", end="\r")
#                 if queue:
#                     queue.put([None, (batch, size_train_dataset+size_validate_dataset), None, None, None])
#         print()

#         # Train on the validation dataset. Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
#         model.train(False)
#         loss = 0
#         with torch.no_grad():
#             for batch, (data, label) in enumerate(validate_dataloader, 1):
#                 data = data.to(device)
#                 label = label.to(device)
#                 output = model(data)
#                 loss += loss_function(output, label.float())
#                 if (batch) % 100 == 0:
#                     print(f"Validating batch {batch}/{size_validate_dataset}...", end="\r")
#                     if queue:
#                         queue.put([None, (size_train_dataset+batch, size_train_dataset+size_validate_dataset), None, None, None])
#         print()
#         loss /= size_validate_dataset
#         validation_losses.append(loss)
#         print(f"Average loss: {loss:,.0f}")

#         # Save the model parameters periodically.
#         if (epoch) % 1 == 0:
#             save(FILEPATH_MODEL, epoch, model, optimizer, [*previous_validation_losses, *validation_losses])
        
#         if queue:
#             queue.put([(epoch, epochs[-1]), None, epochs, validation_losses, previous_validation_losses])
        
#         if queue_from_gui:
#             if not queue_from_gui.empty():
#                 # Stop training.
#                 if queue_from_gui.get() == True:
#                     queue_from_gui.queue.clear()
#                     break
    
#     # Save the model parameters.
#     save(FILEPATH_MODEL, epoch, model, optimizer, [*previous_validation_losses, *validation_losses])
    
#     # Plot the loss history.
#     if not queue:
#         plt.figure()
#         if previous_validation_losses:
#             plt.plot(range(1, epochs[0]), previous_validation_losses, 'o', color=Colors.GRAY_LIGHT)
#         plt.plot(epochs, validation_losses, '-o', color=Colors.BLUE)
#         plt.ylim(bottom=0)
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.grid(axis='y')
#         plt.show()

def main(epoch_count: int, learning_rate: float, batch_size: int, Model: nn.Module, training_split: float, train_existing=None, test_only=False, queue=None, queue_from_gui=None):
    """Train and test the model."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Create the datasets.
    train_dataloader, validate_dataloader, test_dataloader = create_dataloaders(batch_size)
    size_test_dataset = len(test_dataloader)

    # Initialize the models.
    LATENT_SIZE = 100
    GENERATOR_FEATURES = 64
    DISCRIMINATOR_FEATURES = 64

    model_cnn = FullyCnn(input_channels=INPUT_CHANNELS, output_size=LATENT_SIZE)
    model_generator = GanGenerator(input_channels=LATENT_SIZE, number_features=GENERATOR_FEATURES, output_channels=OUTPUT_CHANNELS)
    model_discriminator = GanDiscriminator(number_features=DISCRIMINATOR_FEATURES, output_channels=OUTPUT_CHANNELS)

    model_cnn.to(device)
    model_generator.to(device)
    model_discriminator.to(device)

    model_generator.apply(initialize_weights)
    model_discriminator.apply(initialize_weights)

    # Train on the training and validation datasets.
    if not test_only:
        train_gan(device, model_cnn, model_generator, model_discriminator, learning_rate, epoch_count, train_dataloader, validate_dataloader, train_existing, test_only, queue, queue_from_gui)

    model_cnn.train(False)
    model_generator.train(False)
    model_discriminator.train(False)

    # Test on the testing dataset.
    test_labels = []
    test_outputs = []
    with torch.no_grad():
        for batch, (data, label) in enumerate(test_dataloader, 1):
            data = data.to(device)
            label = label.to(device)

            latent = torch.randn(label.size(0), 100, 1, 1, device=device)  # model_cnn(data)
            test_output = model_generator(latent)
            test_output = test_output[0, :, ...].cpu().detach().numpy()
            label = label[0, :, ...].cpu().numpy()
            test_labels.append(label)
            test_outputs.append(test_output)

            # Scale the generated image and label from [-1, 1] to [0, 255], if training the GAN.
            test_output = (test_output + 1) / 2 * 255
            label = (label + 1) / 2 * 255
            
            # Vertically concatenate the label image and model output and write the combined image to a file.
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
    EPOCHS = 10
    LEARNING_RATE = 1e-7
    BATCH_SIZE = 1
    Model = Nie

    TRAINING_SPLIT = 0.8

    main(EPOCHS, LEARNING_RATE, BATCH_SIZE, Model, TRAINING_SPLIT, train_existing=True, test_only=not True)