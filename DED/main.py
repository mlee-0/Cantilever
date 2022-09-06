'''
Train and test the model.
'''


import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader

from networks import *
from setup import *


# Model parameters file path.
FILEPATH_MODEL = os.path.join(FOLDER_ROOT, 'model.pth')

# Values used to represent real and fake images for the GAN.
LABEL_REAL = 1
LABEL_FAKE = 0


class DedDataset(Dataset):
    def __init__(self, dataset: str, experiment_number: int):
        if experiment_number == 1:
            folder_labels = os.path.join(FOLDER_ROOT, "Exp#1_(sheet#1)")
            total_samples = 81
            sheet_index = 0
            # Number of unique sets of input parameters (product of numbers of unique values for each individual parameter found in the dataset).
            self.embedding_size = 4 * 3 * 4
        elif experiment_number == 2:
            folder_labels = os.path.join(FOLDER_ROOT, "Exp#2_(sheet#2)")
            total_samples = 192
            sheet_index = 1
            self.embedding_size = 4 * 4 * 3

        sample_indices = range(total_samples)
        
        self.inputs = read_inputs("Dataset_experiments_031722.xlsx", sheet_index=sheet_index, sample_indices=sample_indices)

        # Load previously generated label images.
        files = glob.glob(os.path.join(folder_labels, "*.pickle"))
        files = [_ for _ in files if dataset in _]
        if files:
            file = files[0]
            with open(file, "rb") as f:
                self.labels = pickle.load(f)
            print(f"Loaded label images from {file}.")
        # Create label images and save them as a pickle file.
        else:
            self.labels = read_labels(folder_labels, sample_indices=sample_indices)
            file = f"{dataset}_labels.pickle"
            # with open(os.path.join(folder_labels, file), "wb") as f:
            #     pickle.dump(self.labels, f)
            # print(f"Saved label images to {file}.")

        # Perform data augmentation.
        augment_data = True
        if augment_data:
            augment_functions = (lambda label: label[:, :, ::-1], lambda label: label + np.random.normal(0, 0.1, label.shape))
            dataset_size_default = len(self.labels)

            for function in augment_functions:
                augmented_labels = []
                for label in self.labels:
                    augmented_label = function(label.copy())
                    augmented_labels.append(augmented_label)
                
                # Insert the augmented labels and corresponding inputs.
                self.inputs = np.repeat(self.inputs, 2)
                index = 1
                for augmented_label in augmented_labels:
                    self.labels.insert(index, augmented_label)
                    index += 2
                
            print(f"Augmented dataset from size {dataset_size_default} to {len(self.labels)}.")
        
        assert len(self.inputs) == len(self.labels), f"Number of inputs {len(self.inputs)} does not match number of labels {len(self.labels)}"
        self.number_samples = len(self.labels)

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        # Return copies of arrays so that arrays are not modified.
        return np.copy(self.inputs[index]), np.copy(self.labels[index])

class DedDatasetClassification(Dataset):
    def __init__(self) -> None:
        folder = "Classification_CNN_UNIST"
        images_burr = read_labels(os.path.join(FOLDER_ROOT, folder, "burr"))
        images_cracked = read_labels(os.path.join(FOLDER_ROOT, folder, "cracked"))
        images_normal = read_labels(os.path.join(FOLDER_ROOT, folder, "normal"))
        
        self.inputs = images_burr + images_cracked + images_normal
        self.labels = [0] * len(images_burr) + [1] * len(images_cracked) + [2] * len(images_normal)

        # Randomize order of images and labels.
        _ = list(zip(self.inputs, self.labels))
        random.shuffle(_)
        self.inputs, self.labels = zip(*_)
        self.inputs = list(self.inputs)
        self.labels = list(self.labels)
        
        # Data augmentation.
        inputs_augmented = []
        for image in self.inputs:
            inputs_augmented.append(np.copy(image[:, :, ::-1]))
        self.inputs.extend(inputs_augmented)
        self.labels.extend(self.labels)
        
        # Convert labels to one-hot tensors.
        self.labels = nn.functional.one_hot(torch.tensor(np.array(self.labels)), 3)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index, :]


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

    print(f"Loaded model from {filepath} trained for {checkpoint['epoch']} epochs.")
    
    return epoch

def initialize_weights(model: nn.Module):
    """Custom weight initialization."""
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def train_gan(device: str, generator: nn.Module, discriminator: nn.Module, learning_rate: float, epoch_count: int, loss: str, train_dataloader: DataLoader, train_existing: bool, test_only: bool, queue = None, queue_from_gui = None) -> None:
    """Train a network consisting of a CNN and a GAN on the given training and validation datasets."""

    assert loss in ("gan", "wgan", "wgan-gp")

    size_train_dataset = len(train_dataloader)

    # Create the loss functions.
    if loss in ("wgan", "wgan-gp"):
        loss_function_discriminator = lambda real_output, fake_output: -1 * torch.mean(real_output) + 1 * torch.mean(fake_output)
        loss_function_generator = lambda fake_output: -torch.mean(fake_output)
    elif loss == "gan":
        bce = nn.BCELoss()
        loss_function_discriminator = lambda real_output, fake_output: bce(torch.ones_like(real_output), real_output) + bce(torch.zeros_like(fake_output), fake_output)
        loss_function_generator = lambda fake_output: bce(torch.ones_like(fake_output), fake_output)

    # Create the optimizers.
    if loss == "wgan-gp":
        if learning_rate != 1e-4:
            print(f"A learning rate of 1e-4 is recommended for {loss}.")
        optimizer_generator = torch.optim.Adam(params=generator.parameters(), lr=learning_rate, betas=(0.0, 0.999))
        optimizer_discriminator = torch.optim.Adam(params=discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.999))
    elif loss == "wgan":
        if learning_rate != 5e-5:
            print(f"A learning rate of 5e-5 is recommended for {loss}.")
        optimizer_generator = torch.optim.RMSprop(params=generator.parameters(), lr=learning_rate)
        optimizer_discriminator = torch.optim.RMSprop(params=discriminator.parameters(), lr=learning_rate)
    elif loss == "gan":
        optimizer_generator = torch.optim.Adam(params=generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_discriminator = torch.optim.Adam(params=discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Load the model and optimizer if saved previously.
    epoch = 1
    previous_losses_generator = []
    previous_losses_discriminator = []
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
                (generator, discriminator),
                (optimizer_generator, optimizer_discriminator),
                (previous_losses_generator, previous_losses_discriminator),
            )
    else:
        train_existing = False
        test_only = False
    
    epochs = range(epoch, epoch+epoch_count)
    if queue:
        queue.put([(epochs[0]-1, epochs[-1]), None, None, None, None])
    
    # Initialize the validation losses.
    losses_generator = []
    losses_discriminator = []

    # Main training-validation loop.
    for epoch in epochs:
        print(f"\nEpoch {epoch}/{epochs[-1]} ({time.strftime('%I:%M:%S %p')})")

        generator.train(True)
        discriminator.train(True)

        loss_generator = 0
        loss_discriminator = 0

        # Train on the training dataset.
        for batch, (data, real_data) in enumerate(train_dataloader, 1):
            # Add random noise to images for the first # epochs to improve stability of GAN training.
            EPOCHS_RANDOM_NOISE = 50
            if epoch <= EPOCHS_RANDOM_NOISE:
                std = 0.1 * max([1 - epoch/EPOCHS_RANDOM_NOISE, 0])  # Decays linearly to 0.0
                mean = 0.0
                real_data = real_data + torch.randn(real_data.size(), device=device) * std + mean
            
            data = data.to(device)
            real_data = real_data.to(device)
            real_data = real_data.float()

            # # Reset gradients.
            # optimizer_generator.zero_grad()
            # optimizer_discriminator.zero_grad()

            # Number of weight updates for the discriminator for every 1 weight update for the generator.
            if loss in ("wgan", "wgan-gp"):
                d_updates_per_g = 5
            else:
                d_updates_per_g = 1
            
            # generator.requires_grad_(False)
            # discriminator.requires_grad_(True)

            for _ in range(d_updates_per_g):
                # Train the discriminator with real images.
                discriminator.zero_grad()
                real_output = discriminator(real_data, data).view(-1)
                # # Calculate the loss of the discriminator.
                # loss_real = -real_output.mean()
                # # loss_real = loss_function(real_output, label_discriminator)
                # # Calculate gradients.
                # loss_real.backward()

                # Train the discriminator with fake images.
                fake_data = generator(torch.randn(real_data.size(0), 100, 1, 1, device=device), data)
                fake_output = discriminator(fake_data.detach(), data).view(-1)
                # # Calculate the loss of the discriminator.
                # loss_fake = +fake_output.mean()
                # # label_discriminator[:] = LABEL_FAKE
                # # loss_fake = loss_function(fake_output, label_discriminator)
                # # Calculate gradients.
                # loss_fake.backward()

                # Calculate total loss of discriminator by summing real and fake losses.
                loss_current_discriminator = loss_function_discriminator(real_output, fake_output)
                loss_current_discriminator.backward()
                # loss_current_discriminator = loss_real + loss_fake

                # Add gradient penalty for WGAN-GP.
                if loss == "wgan-gp":
                    LAMBDA = 10
                    alpha = torch.rand(real_data.size())
                    interpolates = real_data + alpha * (fake_data - real_data)
                    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
                    discriminator.zero_grad()
                    gp_output = discriminator(interpolates, data)
                    
                    gradients = torch.autograd.grad(outputs=gp_output, inputs=interpolates, grad_outputs=torch.ones(gp_output.size(), device=device), create_graph=True)[0]  # retain_graph=True)[0]
                    gp = LAMBDA * ((gradients.norm(2, dim=[1,2,3]) - 1) ** 2).mean()
                    gp.backward()
                    loss_current_discriminator += gp

                # Update discriminator.
                optimizer_discriminator.step()

                # Clip the weights of the discriminator.
                if loss == "wgan":
                    with torch.no_grad():
                        for parameter in discriminator.parameters():
                            parameter.copy_(torch.clip(parameter, -0.01, +0.01))

            # generator.requires_grad_(True)
            # discriminator.requires_grad_(False)
            
            # Train the generator.
            generator.zero_grad()
            # Forward pass fake images through the discriminator.
            fake_data = generator(torch.randn(real_data.size(0), 100, 1, 1, device=device), data)
            fake_output = discriminator(fake_data, data).view(-1)
            # Calculate the loss of the generator, assuming images are real in order to calculate loss correctly.
            loss_current_generator = loss_function_generator(fake_output)
            # Calculate gradients.
            loss_current_generator.backward()
            # Update generator.
            optimizer_generator.step()

            loss_generator += loss_current_generator.item()
            loss_discriminator += loss_current_discriminator.item()

            # Periodically display progress.
            if batch % 1 == 0:
                print(f"Batch {batch}/{size_train_dataset}: {loss_generator/batch:.2e} (generator), {loss_discriminator/batch:.2e} (discriminator)", end="\r")
                if queue:
                    queue.put([None, (batch, size_train_dataset), None, None, None])
        
        loss_generator /= batch
        loss_discriminator /= batch
        losses_generator.append(loss_generator)
        losses_discriminator.append(loss_discriminator)
        print(f"Average loss: {loss_generator:,.2e} (generator), {loss_discriminator:,.2e} (discriminator)")

        # Save the model parameters periodically.
        if epoch % 1 == 0 or epoch == epochs[-1]:
            save(
                FILEPATH_MODEL,
                epoch,
                [generator, discriminator],
                [optimizer_generator, optimizer_discriminator],
                [
                    [*previous_losses_generator, *losses_generator],
                    [*previous_losses_discriminator, *losses_discriminator],
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
    
    # Plot the loss history.
    if not queue:
        plt.figure()
        if previous_losses_generator and previous_losses_discriminator:
            plt.plot(range(1, epochs[0]), previous_losses_generator, 'o', color=Colors.GRAY_LIGHT)
            plt.plot(range(1, epochs[0]), previous_losses_discriminator, '*', color=Colors.GRAY_LIGHT)
        plt.plot(epochs, losses_generator, '-o', color=Colors.BLUE, label="Generator")
        plt.plot(epochs, losses_discriminator, '-*', color=Colors.ORANGE, label="Discriminator")
        plt.legend()
        # plt.ylim(bottom=0)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(axis='y')
        plt.show()

def train_classifier(Model: nn.Module, learning_rate: float, batch_size: int, epoch_count: int, train_existing: bool, test_only: bool, queue=None, queue_from_gui=None) -> None:
    """Train a model on the given datasets."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    k_folds = 5

    dataset = DedDatasetClassification()

    test_accuracies = []
    test_losses = []
    for k in range(k_folds):
        validate_indices = range(k, len(dataset), 10)
        test_indices = range(k+1, len(dataset), 10)
        train_indices = [_ for _ in range(len(dataset)) if _ not in {*validate_indices, *test_indices}]
        train_dataloader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
        validate_dataloader = DataLoader(Subset(dataset, validate_indices), batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=True)

        # Initialize the optimizer and loss function.
        model = Model(1, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()

        # Load their parameters if they have been saved previously.
        epoch = 1
        previous_training_accuracies = []
        previous_validation_accuracies = []
        previous_training_losses = []
        previous_validation_losses = []
        if os.path.exists(FILEPATH_MODEL):
            if not test_only:
                if train_existing is None:
                    train_existing = input(f'Continue training the model in {FILEPATH_MODEL}? [y/n] ') == 'y'
            else:
                train_existing = True
            
            if train_existing:
                epoch = load(FILEPATH_MODEL, device, [model], [optimizer], [previous_validation_losses])
        else:
            train_existing = False
            test_only = False
        
        epochs = range(epoch, epoch+epoch_count)
        if queue:
            queue.put([(epochs[0]-1, epochs[-1]), None, None, None, None])
        
        training_accuracies = []
        validation_accuracies = []
        training_losses = []
        validation_losses = []
        for epoch in epochs:
            print(f"Epoch {epoch} (k = {k})\n------------------------")
            
            model.train(True)

            # Train on the training dataset.
            correct = 0
            loss = 0
            for batch, (data, label) in enumerate(train_dataloader, 1):
                data = data.to(device)
                label = label.to(device)
                output = model(data)

                correct += sum(torch.argmax(output, dim=1) == torch.argmax(label, dim=1))
                loss_current = loss_function(output, label.float())
                loss += loss_current.item()
                # Reset gradients of model parameters.
                optimizer.zero_grad()
                # Backpropagate the prediction loss.
                loss_current.backward()
                # Adjust model parameters.
                optimizer.step()

                if batch % 5 == 0:
                    print(f"Training batch {batch}/{len(train_dataloader)}...", end="\r")
                    if queue:
                        queue.put([None, (batch, len(train_dataloader)+len(validate_dataloader)), None, None, None])
            print()
            accuracy = correct / len(train_indices)
            loss /= len(train_dataloader)
            training_accuracies.append(accuracy)
            training_losses.append(loss)
            print(f"Average training accuracy: {100*accuracy:,.1f}%")
            print(f"Average training loss: {loss:,.1f}")

            model.train(False)

            # Train on the validation dataset. Set model to evaluation mode, which is required if it contains batch normalization layers, dropout layers, and other layers that behave differently during training and evaluation.
            correct = 0
            loss = 0
            with torch.no_grad():
                for batch, (data, label) in enumerate(validate_dataloader, 1):
                    data = data.to(device)
                    label = label.to(device)
                    output = model(data)

                    correct += sum(torch.argmax(output, dim=1) == torch.argmax(label, dim=1))
                    loss += loss_function(output, label.float()).item()
                    if batch % 5 == 0:
                        print(f"Validating batch {batch}/{len(validate_dataloader)}...", end="\r")
                        if queue:
                            queue.put([None, (len(train_dataloader)+batch, len(train_dataloader)+len(validate_dataloader)), None, None, None])
            print()
            accuracy = correct / len(validate_indices)
            loss /= len(validate_dataloader)
            validation_accuracies.append(accuracy)
            validation_losses.append(loss)
            print(f"Average validation accuracy: {100*accuracy:,.1f}%")
            print(f"Average validation loss: {loss:,.1f}")

            # # Save the model parameters periodically.
            # if epoch % 1 == 0:
            #     save(FILEPATH_MODEL, epoch, model, optimizer, [], [*previous_validation_losses, *validation_losses])
            
            if queue:
                queue.put([(epoch, epochs[-1]), None, epochs, validation_losses, previous_validation_losses])
            
            if queue_from_gui:
                if not queue_from_gui.empty():
                    # Stop training.
                    if queue_from_gui.get() == True:
                        queue_from_gui.queue.clear()
                        break
        
        # # Save the model parameters.
        # save(FILEPATH_MODEL, epoch, model, optimizer, [*previous_validation_losses, *validation_losses])

        # Test on the testing dataset.
        model.train(False)
        correct = 0
        loss = 0
        with torch.no_grad():
            for batch, (data, label) in enumerate(test_dataloader, 1):
                data = data.to(device)
                label = label.to(device)
                output = model(data)

                correct_current = sum(torch.argmax(output, dim=1) == torch.argmax(label, dim=1))
                correct += correct_current
                loss += loss_function(output, label.float()).item()
                print(f"Batch {batch}: predicted {output}, true {label}")
        print()
        accuracy = correct / len(test_indices)
        loss /= len(test_dataloader)
        print(f"Average testing accuracy: {100*accuracy:,.1f}%")
        print(f"Average testing loss: {loss:,.1f}")
        test_accuracies.append(accuracy)
        test_losses.append(loss)
        
        # Plot the accuracy and loss history.
        if not queue:
            plt.figure()

            plt.subplot(1, 2, 1)
            if previous_training_accuracies:
                plt.plot(range(1, epochs[0]), previous_training_accuracies, '.:', color=Colors.GRAY_LIGHT)
            if previous_validation_accuracies:
                plt.plot(range(1, epochs[0]), previous_validation_accuracies, '.-', color=Colors.GRAY_LIGHT)
            plt.plot(epochs, training_accuracies, '.:', color=Colors.ORANGE, label="Training")
            plt.plot(epochs, validation_accuracies, '.-', color=Colors.BLUE, label="Validation")
            plt.legend()
            plt.ylim(bottom=0)
            plt.grid(axis='y')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Accuracy")

            plt.subplot(1, 2, 2)
            if previous_validation_losses:
                plt.plot(range(1, epochs[0]), previous_training_losses, '.:', color=Colors.GRAY_LIGHT)
                plt.plot(range(1, epochs[0]), previous_validation_losses, '.-', color=Colors.GRAY_LIGHT)
            plt.plot(epochs, training_losses, '.:', color=Colors.ORANGE, label="Training")
            plt.plot(epochs, validation_losses, '.-', color=Colors.BLUE, label="Validation")
            plt.legend()
            plt.ylim(bottom=0)
            plt.grid(axis='y')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss")

            plt.show()
    
    print(f"Average testing accuracy over {k_folds} folds: {100 * np.mean(test_accuracies)}%")
    print(f"Average testing loss over {k_folds} folds: {np.mean(test_losses)}")

def main(experiment_number: int, epoch_count: int, learning_rate: float, batch_size: int, loss: str, model_size: int, train_existing=None, test_only=False, queue=None, queue_from_gui=None):
    """Train and test the model."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Create the datasets.
    train_dataset = DedDataset("train", experiment_number)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the models.
    LATENT_SIZE = 100

    generator = GanGenerator(input_channels=LATENT_SIZE, number_features=model_size, output_channels=OUTPUT_CHANNELS)  #, embedding_size=train_dataset.embedding_size)
    discriminator = GanDiscriminator(number_features=model_size, output_channels=OUTPUT_CHANNELS)  #, embedding_size=train_dataset.embedding_size)

    generator.to(device)
    discriminator.to(device)

    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    # Train on the training and validation datasets.
    if not test_only:
        train_gan(device, generator, discriminator, learning_rate, epoch_count, loss, train_dataloader, train_existing, test_only, queue, queue_from_gui)

    generator.train(False)
    discriminator.train(False)

    # Test by generating images for the specified classes.
    test_outputs = []
    with torch.no_grad():
        SAMPLES_PER_CLASS = 1
        classes = np.arange(train_dataset.embedding_size)
        classes = np.repeat(classes, SAMPLES_PER_CLASS)
        for batch, data in enumerate(classes, 1):
            data = torch.tensor([data], dtype=int, device=device)
            latent = torch.randn(1, LATENT_SIZE, 1, 1, device=device)

            test_output = generator(latent, data)
            test_output = test_output[0, :, ...].cpu().detach().numpy()
            test_outputs.append(test_output)

            # Scale the generated image and label from [-1, 1] to [0, 255], if training the GAN.
            test_output = (test_output + 1) / 2 * 255

            # Add channels if it only has 1 channel.
            if test_output.shape[0] == 1:
                test_output = np.concatenate([test_output] * 3, axis=0)
            
            # Write the image to a file.
            write_image(
                test_output.transpose((1, 2, 0)),
                os.path.join(FOLDER_RESULTS, f'{batch:03d}_({data.item()}).png'),
            )
            
            if queue:
                queue.put([None, (batch, len(classes)), None, None, None])
    
    print(f"Wrote {batch} test images in {FOLDER_RESULTS}.")


if __name__ == '__main__':
    kwargs = {
        "experiment_number": 2,

        "epoch_count": 10,
        "learning_rate": 1e-4,
        "batch_size": 8,
        "loss": "wgan-gp",
        "model_size": 64,

        "train_existing": not True,
        "test_only": False,
    }

    main(**kwargs)
    # train_classifier(Model=ClassifierCnn, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_count=EPOCHS, train_existing=False, test_only=False)