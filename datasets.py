import torch
from torch.utils.data import Dataset

from helpers import *


class CantileverDataset(Dataset):
    """
    Dataset that contains 4D input images and label images. Generates input images and loads a .pickle file of pre-generated label images.

    Input images have shape (batch, channel, height, length).
    Label images have shape (batch, channel, height, length).
    """

    def __init__(self, is_3d: bool, normalize_inputs: bool=False, transformation_exponent: float=None, transformation_logarithm: Tuple[float, float]=None):
        self.transformation_exponent = transformation_exponent
        self.transformation_logarithm = transformation_logarithm

        samples = read_samples(os.path.join(FOLDER_ROOT, "samples.csv"))

        if is_3d:
            folder_labels = os.path.join(FOLDER_ROOT, "Labels 3D")
            filename_labels = "labels.pickle"
        else:
            folder_labels = os.path.join(FOLDER_ROOT, "Labels 2D")
            filename_labels = "labels.pickle"
        
        # Load previously generated labels.
        self.labels = read_pickle(os.path.join(folder_labels, filename_labels))
        self.labels = torch.tensor(self.labels)

        # The raw maximum value found in the entire dataset.
        self.max_value = self.labels.max()
        # The maximum value to scale the label data to after applying the transformation.
        self.scaled = 100

        # Apply the transformation to the label data.
        if self.transformation_exponent is not None:
            self.labels_transformed = self.transform_exponentiation(self.labels)
            self.transform, self.untransform = self.transform_exponentiation, self.untransform_exponentiation
        elif self.transformation_logarithm is not None:
            self.labels_transformed = self.transform_logarithm(self.labels)
            self.transform, self.untransform = self.transform_logarithm, self.untransform_logarithm
        else:
            raise Exception("For no transform, use an exponentiation transform with an exponent of 1.")

        # # Show the histogram of the original data and transformed data, both of which have the same range of values.
        # plt.figure()
        # plt.hist((self.labels / self.labels.max()).numpy().flatten(), bins=100, alpha=0.5)
        # plt.hist((self.labels_transformed / self.labels_transformed.max()).numpy().flatten(), bins=100, alpha=0.5)
        # plt.show()

        # Create input images.
        self.inputs = generate_input_images(samples, is_3d=is_3d)
        self.inputs = torch.tensor(self.inputs).float()
        if normalize_inputs:
            # Normalize to zero mean and unit standard deviation.
            self.inputs -= self.inputs.mean()
            self.inputs /= self.inputs.std()

        # # Visualize input and label data.
        # import random
        # plt.figure()
        # for i in random.sample(range(self.inputs.size(0)), k=3):
        #     for channel in range(self.inputs.size(1)):
        #         plt.subplot(1, self.inputs.size(1), channel+1)
        #         plt.imshow(self.inputs[i, channel, ...], cmap='gray', vmin=0, vmax=self.inputs.max())
        #     plt.show()
        # plt.figure()
        # import random
        # for i in range(3):
        #     plt.subplot(1, 3, i+1)
        #     plt.imshow(self.labels[random.randint(0, 999), 0, ...])
        # plt.show()

        # Print information about the data.
        print(f"\nDataset '{type(self)}':")
        
        print(f"Input data:")
        print(f"\tShape: {self.inputs.size()}")
        print(f"\tMemory: {self.inputs.storage().nbytes()/1e6:,.2f} MB")
        print(f"\tMin, max: {self.inputs.min()}, {self.inputs.max()}")

        print(f"Label data:")
        print(f"\tShape: {self.labels.size()}")
        print(f"\tMemory: {self.labels.storage().nbytes()/1e6:,.2f} MB")
        print(f"\tExponentiation transformation: {self.transformation_exponent}")
        print(f"\tLogarithm transformation: {self.transformation_logarithm}")
        print(f"\tMin, max: {self.labels_transformed.min()}, {self.labels_transformed.max()}")
        print(f"\tMean: {self.labels_transformed.mean()}")
        print(f"\tOriginal max: {self.max_value}")

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, index):
        """Return input and label images."""
        return self.inputs[index, ...], self.labels_transformed[index, ...]

    def transform_exponentiation(self, y: torch.Tensor):
        """Scale data to [0, 1], exponentiate, and scale to the specified range."""
        return (y / self.max_value) ** self.transformation_exponent * self.scaled

    def untransform_exponentiation(self, y: torch.Tensor):
        return ((y / self.scaled) ** (1 / self.transformation_exponent)) * self.max_value

    def transform_logarithm(self, y: torch.Tensor):
        """Scale data to [x[0], x[1]], take the natural logarithm, and scale to the specified range."""
        x_1, x_2 = self.transformation_logarithm
        y = y / self.max_value
        y = y * (x_2 - x_1) + x_1
        y = np.log(y)
        self._min = y.min()
        y -= self._min
        self._max = y.max()
        y /= self._max
        y *= self.scaled
        return y

    def untransform_logarithm(self, y: torch.Tensor):
        x_1, x_2 = self.transformation_logarithm
        y = y / self.scaled
        y = y * self._max
        y = y + self._min
        y = np.exp(y)
        y = (y - x_1) / (x_2 - x_1)
        y = y * self.max_value
        return y

class CantileverDataset3d(Dataset):
    """
    Dataset that contains 5D input images and label images for use with 3D convolution. Generates input images and loads a .pickle file of pre-generated label images.

    Input images have shape (batch, channel, height, length, width).
    Label images have shape (batch, channel=1, height, length, width).
    """
    def __init__(self, samples: pd.DataFrame, normalize_inputs: bool=False, transformation_exponent: float=1):
        self.number_samples = len(samples)
        self.transformation_exponent = transformation_exponent
        print(f"Using transformation exponent: {self.transformation_exponent}.")

        folder_labels = os.path.join(FOLDER_ROOT, "Labels 3D")
        
        # Load previously generated label images.
        self.labels = read_pickle(os.path.join(folder_labels, "labels.pickle"))
        # Transpose dimensions for shape: (samples, 1, height (Y), length (X), width (Z)).
        self.labels = np.expand_dims(self.labels, axis=1).transpose((0, 1, 3, 4, 2))
        print(f"Label images take up {self.labels.nbytes/1e6:,.2f} MB.")

        # The maximum value found in the entire dataset.
        self.max_value = np.max(self.labels)

        # Apply the transformation to the label values.
        self.labels = self.transform(self.labels, inverse=False)
        
        # Create input images.
        self.inputs = generate_input_images_3d(samples)
        if normalize_inputs:
            raise NotImplementedError()
        print(f"Input images take up {self.inputs.nbytes/1e6:,.2f} MB.")

        # Numerical inputs, scaled to [0, 1].
        self.loads = (samples[load.name] - load.low) / (load.high - load.low)

        # Number of channels in input and label images.
        self.input_channels = self.inputs.shape[1]
        self.output_channels = self.labels.shape[1]

    def __len__(self):
        return self.number_samples
    
    def __getitem__(self, index):
        """Return input and label images."""
        # Return copies of arrays so that arrays are not modified.
        return np.copy(self.inputs[index, ...]), np.copy(self.labels[index, ...])
    
    def transform(self, y: np.ndarray, inverse=False) -> np.ndarray:
        if not inverse:
            return y ** self.transformation_exponent
        else:
            return y ** (1 / self.transformation_exponent)