# Cantilever
2D stress distribution prediction in cantilever beams using convolutional neural network.

Variable parameters:
* Beam length
* Beam height
* Load angle
* Load location

Input images:
* Channel 1: Beam shape and load location
* Channel 2: Load angle

Output images:
* 2D von Mises stress distributions generated by FEA simulations in ANSYS (Mechanical APDL)

The 2023-02 dataset and 2023-05 dataset were both generated using the same simulation parameters, but 2023-05 was generated using an updated and simplified Ansys script, which had minor changes to syntax and calculations and small discrepancies in values. Future training should be done using the 2023-05 dataset.


## Creating the Dataset
Ansys (Mechanical APDL) is used to run simulations and generate response data, which are then processed in Python.

Define the simulation parameters in the script `ansys_script.lgw`. The script loops over each combination of simulation parameters, and it simulates responses and saves the response data to text files. Refer to [this reference](https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_cmd/Hlp_C_CmdTOC.html) on Ansys APDL commands.

Open Ansys and run the script `ansys_script.lgw`.
* File > Change Directory... > [Select the folder where the script exists]
* File > Read Input from... > [Select the script]
* To maximize performance, close the File Explorer window of the folder in which the program is running while simulations are running.

Each line in a text file contains the response value at a different node (not an element) along with the X, Y, and Z coordinates of that node. The coordinates of the node are used later to insert each response value into the proper location in a matrix.

Run `preprocessing.py` to process and cache the text files for easier loading when training models. Specifically, this code reads the response data contained in the text files and converts them into tensors and saves them as [`.pickle` files](https://docs.python.org/3/library/pickle.html). Caching the data in this way avoids having to read and convert the text files directly every time a model is trained, which takes longer.

Define the location and name of the `.pickle` file in the `CantileverDataset` class in `datasets.py`.


## Training and Testing
Call the `main()` function in `main.py` to train and test the model. The model architecture is defined in `networks.py`. Results and model predictions are plotted after testing.

Set `train` to `True` to train the model, and set `test` to `True` to test the model. When training a model from scratch, `train_existing` must be set to `False`. Manually specify training hyperparameters, such as number of epochs, learning rate, batch size, etc.

During training, the best model weights (determined by lowest validation loss) are tracked and saved separately.


## Google Colab

### Uploading Files
1. Upload all required `.py` files other than `main.py` to the current directory: `/content/`. These files need to be uploaded each time the session resets.
2. Uploaded all other required files (`.csv`, `.txt`, etc.) to a folder of choice in Google Drive. This folder is specified in the code as: `drive/My Drive/<FOLDER>`. Click "Mount Drive" in the sidebar to make files from Google Drive available. This creates a folder called `drive` in the directory `/content/`.

### Using GPU
To use GPU, click Runtime > Change runtime type.