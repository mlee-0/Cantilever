"""Define classes that load datasets."""


from typing import *

import torch
from torch.utils.data import Dataset

from preprocessing import *


def transform_exponentiation(data: torch.Tensor, power: float, inverse: bool):
    """Raise the data to a power. The data is assumed to already be in the range [0, 1]."""

    if not inverse:
        data = data ** power
    else:
        data = data ** (1/power)

    return data

def transform_logarithmic(data: torch.Tensor, input_range: Tuple[float, float], inverse: bool):
    """Scale the data to a range and then apply the natural logarithm. The data is assumed to already be in the range [0, 1]."""

    x_1, x_2 = input_range

    if not inverse:
        data = data * (x_2 - x_1) + x_1
        data = np.log(data)
    else:
        data = np.exp(data)
        data = (data - x_1) / (x_2 - x_1)

    return data


class CantileverDataset(Dataset):
    """
    Load a stress distribution dataset obtained in FEA. Contains 2-channel input images and 1-channel label images. Generates input images on initialization and loads a .pickle file of preprocessed labels. Optionally specify a transformation to apply on the labels.

    Inputs:
    `normalize_inputs`: Normalize the input data to have zero mean and unit variance. Not recommended.
    `transformation_exponentiation`: A power to which the labels are raised. Use None for no transformation.
    `transformation_logarithmic`: A tuple defining the range to which the labels are scaled, before the natural logarithm is applied. Use None for no transformation.
    `label_max`: The maximum value to which the labels are scaled after applying any transformations. Use None for no scaling.
    """

    def __init__(self, normalize_inputs: bool=False, transformation_exponentiation: float=None, transformation_logarithmic: Tuple[float, float]=None, label_max: float=100):
        self.label_max = label_max

        parameters = generate_simulation_parameters()

        # Create input images.
        self.inputs = make_inputs(parameters)
        self.inputs = torch.tensor(self.inputs).float()
        # Normalize to zero mean and unit standard deviation.
        if normalize_inputs:
            self.inputs -= self.inputs.mean()
            self.inputs /= self.inputs.std()

        # Load preprocessed labels.
        self.labels = read_pickle(os.path.join('Stress 2D 2023-05', 'labels.pickle'))
        self.labels = torch.tensor(self.labels)

        # The raw maximum value found in the entire dataset.
        self.label_max_raw = self.labels.max()

        # Define the transformation and its inverse.
        if transformation_exponentiation is not None:
            self.transformation, self.transformation_parameter = transform_exponentiation, transformation_exponentiation
        elif transformation_logarithmic is not None:
            self.transformation, self.transformation_parameter = transform_logarithmic, transformation_logarithmic
        else:
            # Raise to a power of 1 for no transformation.
            self.transformation, self.transformation_parameter = transform_exponentiation, 1

        # Transform the labels.
        self.labels_transformed = self.transform(self.labels)

        # Print information about the data.
        print(f"\nDataset '{type(self)}':")
        
        print(f"Input data:")
        print(f"\tShape: {self.inputs.size()}")
        print(f"\tMemory: {self.inputs.storage().nbytes()/1e6:,.2f} MB")
        print(f"\tMin, max: {self.inputs.min()}, {self.inputs.max()}")

        print(f"Label data:")
        print(f"\tShape: {self.labels.size()}")
        print(f"\tMemory: {self.labels.storage().nbytes()/1e6:,.2f} MB")
        print(f"\tExponentiation transformation: {transformation_exponentiation}")
        print(f"\tLogarithmic transformation: {transformation_logarithmic}")
        print(f"\tMin, max: {self.labels_transformed.min()}, {self.labels_transformed.max()}")
        print(f"\tMean: {self.labels_transformed.mean()}")
        print(f"\tOriginal max: {self.label_max_raw}")

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, index):
        """Return input and label images."""
        return self.inputs[index, ...], self.labels_transformed[index, ...]

    def transform(self, y: torch.Tensor):
        # Scale to [0, 1].
        y = y / self.label_max_raw

        # Transform the data and store the resulting minimum and maximum values.
        y = self.transformation(y, self.transformation_parameter, inverse=False)
        self._min, self._max = y.min(), y.max()

        # Scale to [0, 1].
        y = y - self._min
        y = y / (self._max - self._min)

        # Scale the data to have the specified maximum.
        if self.label_max is not None:
            y = y * self.label_max

        return y
    
    def untransform(self, y: torch.Tensor):
        if self.label_max is not None:
            y = y / self.label_max

        y = y * (self._max - self._min)
        y = y + self._min

        y = self.transformation(y, self.transformation_parameter, inverse=True)

        y = y * self.label_max_raw

        return y

class CantileverDataset3d(Dataset):
    """
    Dataset of inputs and labels for 3D stress predictions. Generates inputs on intialization and loads a .pickle file of preprocessed labels.
    """

    def __init__(self, normalize_inputs: bool=False, transformation_exponent: float=1):
        parameters = generate_simulation_parameters()

        self.transformation_exponent = transformation_exponent
        print(f"Using transformation exponent: {self.transformation_exponent}.")

        # Load preprocessed labels.
        self.labels = read_pickle(os.path.join('Labels 3D', 'labels.pickle'))
        # Transpose dimensions for shape: (samples, 1, height (Y), length (X), width (Z)).
        self.labels = np.expand_dims(self.labels, axis=1).transpose((0, 1, 3, 4, 2))
        print(f"Label images take up {self.labels.nbytes/1e6:,.2f} MB.")

        # The maximum value found in the entire dataset.
        self.max_value = np.max(self.labels)

        # Apply the transformation to the label values.
        self.labels = self.transform(self.labels, inverse=False)
        
        # Create inputs.
        self.inputs = make_inputs_3d(parameters)
        if normalize_inputs:
            raise NotImplementedError()
        print(f"Input images take up {self.inputs.nbytes/1e6:,.2f} MB.")

        # Number of channels in input and label images.
        self.input_channels = self.inputs.shape[1]
        self.output_channels = self.labels.shape[1]

    def __len__(self):
        return self.inputs.size(0)
    
    def __getitem__(self, index):
        """Return input and label images."""
        # Return copies of arrays so that arrays are not modified.
        return np.copy(self.inputs[index, ...]), np.copy(self.labels[index, ...])
    
    def transform(self, y: np.ndarray, inverse=False) -> np.ndarray:
        if not inverse:
            return y ** self.transformation_exponent
        else:
            return y ** (1 / self.transformation_exponent)


if __name__ == '__main__':
    pass