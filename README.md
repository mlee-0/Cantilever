# Thesis


# Cantilever
This program trains a convolutional neural network that predicts the stress distribution in a cantilever beam subjected to a point load at the free end with the following variable parameters:
- Load magnitude
- Load direction
- Geometry (length and height)
- Elastic modulus (constant throughout cantilever)

The input images that represent these parameters and are used as inputs to the model during training contain the following channels:
1. Load magnitude and direction: A black image with a white line. The brightness of the line represents the load magnitude, and the orientation of the line represents the load angle. The counterclockwise positive convention is followed. An angle of 0 degrees points right, and an angle of 90 degrees points up.
2. Geometry: A black background with a white region representing the shape of the cantilever.
3. Elastic modulus: A black background with a white region representing the elastic modulus values. The brightness of the region represents the magnitude of the elastic modulus.

The output images used as labels during training are created from stress distribution data generated by FEA simulations in ANSYS (Mechanical APDL).

## Dataset Generation
Run the following scripts to generate datasets as `.txt` files and ANSYS code as `.lgw` files that are used to automate FEA simulations:
* `generate_train_dataset.py` generates:
    * `samples_train.txt`
    * `ansys_script_train.lgw`
* `generate_validation_dataset.py` generates:
    * `samples_validation.txt`
    * `ansys_script_validation.lgw`
* `generate_test_dataset.py` generates:
    * `samples_test.txt`
    * `ansys_script_test.lgw`

The ANSYS scripts are generated by filling in placeholder text in the template script `ansys_template.lgw`. Changes to the format of the scripts should only be made in this file.

Place the `.txt` files generated by ANSYS in the respective folders:
* `Train Labels`
* `Validation Labels`
* `Test Labels`

### Stratified Sampling
To ensure a balanced dataset used during training, the code selects a subset of the dataset that contains a uniform distribution of stress values. The model is trained only on this subset, and samples not in this subset are never used.

The code creates a histogram of the maximum stress values in each sample, with a specified number of bins that range from 0 to the highest stress found in the entire dataset. An equal number of samples is selected from each bin. For example, to create a dataset with 100 samples using 20 bins, 5 samples are selected from each bin.

The number of bins is manually selected. A larger number of bins results in a more uniform distribution, but it may result in bins that do not have enough samples. A smaller number of bins reduces the possibility of not having enough samples in each bin, but it may result in a less uniform distribution.

The dataset from which the subset is selected should be large enough to provide enough samples in each bin. It should ideally be as uniformly distributed as possible.

### ANSYS (Mechanical APDL)
To run a script in ANSYS:
* File > Change Directory... > [Select the folder containing the `.lgw` files]
* File > Read Input from... > [Select the `.lgw` file]

To run a script in ANSYS from Command Prompt, which provides better performance:
* `cd <path to desired folder>`
    * This changes the current folder.
* `"C:\Program Files\ANSYS Inc\v211\ansys\bin\winx64\MAPDL.exe" -b -i "<name of .lgw file>" -o "out.txt"`
    * `-i` specifies the input file to read from.
    * `-o` specifies the output file in which information is written, which can be ignored.

Simulations:
* Element type: PLANE182 (2D 4-node structural solid)
    * Thickness: 1
* Material properties:
    * Poisson's ratio: 0.3

## Training and Testing
Run `main.py` to train and test the model. Set the number of epochs and learning rate manually. After training, output images generated by the model and the corresponding label images are saved in the current folder. Plots showing the loss history and evaluation metrics are displayed.
* Input images are generated at runtime and are not saved as files.
* Labels generated from ANSYS must be located in the respective folder. Images are generated from this data at runtime.
* Model parameters are saved periodically as `model.pth`.
* Training can be continued where it left off if an existing `model.pth` exists. When prompted, confirm this by entering "y" or "n". To train from scratch, remove `model.pth` from the folder before running.
    * The model being trained must match the model stored in `model.pth`.


## Google Colab

### Uploading Files
1. Upload all required `.py` files other than `main.py` to the current directory: `/content/`. These files need to be uploaded each time the session resets.
2. Uploaded all other required files (`.txt`, etc.) to a folder of choice in Google Drive. This folder is specified in the code as: `drive/My Drive/<FOLDER>`. Click "Mount Drive" in the sidebar to make files from Google Drive available. This creates a folder called `drive` in the directory `/content/`.

### Using GPU
To use GPU, click Runtime > Change runtime type.