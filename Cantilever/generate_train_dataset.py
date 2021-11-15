import colorsys
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from main import load, angle, length, height, ANGLE_PEAKS, NUMBER_DIVISIONS, OUTPUT_SIZE, FOLDER_ROOT, FOLDER_TRAIN_INPUTS, FOLDER_TRAIN_OUTPUTS, NUMBER_DIGITS, generate_input_images


# Dataset size.
NUMBER_SAMPLES = 100
assert NUMBER_SAMPLES % 4 == 0, 'Sample size must be divisible by 4 for angles to be generated properly.'

# Names of files to be generated.
FILENAME_SAMPLES = 'cantilever_samples.txt'
FILENAME_ANSYS_TEMPLATE = 'ansys_template.lgw'
FILENAME_ANSYS_SCRIPT = 'ansys_script.lgw'
FILENAME_STRESS_FEA = 'stress.csv'

# Generate and store sample values for each parameter.
def generate_samples(show_histogram=False):
    # Helper function for generating unevenly spaced samples within a defined range. Setting "increasing" to True makes spacings increase as values increase.
    generate_logspace_samples = lambda low, high, increasing: (((
        np.logspace(
            0, 1, round(NUMBER_SAMPLES/4)+1
            ) - 1) / 9) * (1 if increasing else -1) + (0 if increasing else 1)
        ) * (high - low) + low
    # Generate samples.
    load_samples = np.linspace(load.low, load.high, NUMBER_SAMPLES)
    angle_samples = np.concatenate((
        generate_logspace_samples(angle.low, (ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2, increasing=True)[:-1],
        generate_logspace_samples((ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2, ANGLE_PEAKS[1], increasing=False)[1:],
        generate_logspace_samples(ANGLE_PEAKS[1], (ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2 + ANGLE_PEAKS[1], increasing=True)[:-1],
        generate_logspace_samples((ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2 + ANGLE_PEAKS[1], angle.high, increasing=False)[1:],
        ))
    length_samples = np.linspace(length.low, length.high, NUMBER_SAMPLES)
    height_samples = np.linspace(height.low, height.high, NUMBER_SAMPLES)
    # Randomize ordering of samples.
    np.random.shuffle(load_samples)
    np.random.shuffle(angle_samples)
    np.random.shuffle(length_samples)
    np.random.shuffle(height_samples)
    # Round samples to a fixed number of decimal places.
    load_samples = np.round(load_samples, load.precision)
    angle_samples = np.round(angle_samples, angle.precision)
    length_samples = np.round(length_samples, length.precision)
    height_samples = np.round(height_samples, height.precision)
    
    # Plot histograms for angle samples.
    if show_histogram:
        plt.figure()
        plt.hist(
            angle_samples,
            bins=round(angle.high-angle.low),
            rwidth=0.75,
            color='#0095ff',
            )
        plt.xticks(np.linspace(angle.low, angle.high, 5))
        plt.title(angle.name)
        plt.show()
    return load_samples, angle_samples, length_samples, height_samples

# Write the specified sample values to a text file, and calculate and return the load components to be used in FEA.
def write_samples(samples):
    # Determine the x and y components of the load.
    load_components = []
    for load_sample, angle_sample in zip(*samples[0:2]):
        angle_sample *= (np.pi / 180)
        load_components.append((
            np.round(np.cos(angle_sample) * load_sample / (NUMBER_DIVISIONS[1]), NUMBER_DIGITS),
            np.round(np.sin(angle_sample) * load_sample / (NUMBER_DIVISIONS[1]), NUMBER_DIGITS),
            ))
    # Write samples to text file.
    text = [
        f'{str(i+1).zfill(NUMBER_DIGITS)},  Load: {load_sample:>10},  X load: {load_x:>15},  Y load: {load_y:>15},  Angle: {angle_sample:>10},  Length: {length_sample:>5},  Height: {height_sample:>5}\n'
        for i, (load_sample, angle_sample, length_sample, height_sample, (load_x, load_y)) in enumerate(zip(*samples, load_components))
        ]
    with open(os.path.join(FOLDER_ROOT, FILENAME_SAMPLES), 'w') as file:
        file.writelines(text)
    print(f'Wrote samples in {FILENAME_SAMPLES}.')
    return load_components

# Return the sample values found in the text file previously generated.
def read_samples():
    load_samples, angle_samples, length_samples, height_samples = [], [], [], []
    filename = os.path.join(FOLDER_ROOT, FILENAME_SAMPLES)
    try:
        with open(filename, 'r') as file:
            for line in file.readlines():
                load, *_, angle, length, height = [float(line.split(':')[1]) for line in line.split(',')[1:]]
                load_samples.append(load)
                angle_samples.append(angle)
                length_samples.append(length)
                height_samples.append(height)
    except FileNotFoundError:
        print(f'"{filename}" not found.')
        return None
    else:
        print(f'Found samples in {filename}.')
        return load_samples, angle_samples, length_samples, height_samples

# Write a text file containing ANSYS commands used to perform FEA and generate stress contour images.
def write_ansys_script(samples, load_components):
    # Read the template script.
    with open(os.path.join(FOLDER_ROOT, FILENAME_ANSYS_TEMPLATE), 'r') as file:
        lines = file.readlines()
    
    # Replace placeholder lines in the template script.
    with open(os.path.join(FOLDER_ROOT, FILENAME_ANSYS_SCRIPT), 'w') as file:
        # Initialize dictionary of placeholder strings (keys) and strings they should be replaced with (values).
        placeholder_substitutions = {}
        # Define names of variables.
        loop_variable = 'i'
        samples_variable = 'samples'
        # Add loop commands.
        placeholder_substitutions['! placeholder_loop_start\n'] = f'*DO,{loop_variable},1,{NUMBER_SAMPLES},1\n'
        placeholder_substitutions['! placeholder_loop_end\n'] = f'*ENDDO\n'
        # Add commands that define the array containing generated samples.
        commands_define_samples = [f'*DIM,{samples_variable},ARRAY,{6},{NUMBER_SAMPLES}\n']
        for i in range(NUMBER_SAMPLES):
            commands_define_samples.append(
                f'{samples_variable}(1,{i+1}) = {samples[0][i]},{load_components[i][0]},{load_components[i][1]},{samples[1][i]},{samples[2][i]},{samples[3][i]}\n'
                )
        placeholder_substitutions['! placeholder_define_samples\n'] = commands_define_samples
        # Add meshing commands.
        placeholder_substitutions['! placeholder_mesh_length\n'] = f'LESIZE,_Y1, , ,{NUMBER_DIVISIONS[0]}, , , , ,1\n'
        placeholder_substitutions['! placeholder_mesh_height\n'] = f'LESIZE,_Y1, , ,{NUMBER_DIVISIONS[1]}, , , , ,1\n'
        # Add commands that format and create the output files.
        placeholder_substitutions['! placeholder_define_suffix\n'] = f'suffix = \'{"0"*NUMBER_DIGITS}\'\n'
        placeholder_substitutions['! placeholder_define_number\n'] = f'number = CHRVAL({loop_variable})\n'
        placeholder_substitutions['! placeholder_define_filename\n'] = f'filename = \'stress_%STRFILL(suffix,number,{NUMBER_DIGITS}-STRLENG(number)+1)%\'\n'
        # Substitute all commands into placeholders.
        for placeholder in placeholder_substitutions:
            command = placeholder_substitutions[placeholder]
            indices = [i for i, line in enumerate(lines) if line == placeholder]
            if isinstance(command, list):
                for index in indices:
                    for sub_command in command[::-1]:
                        lines.insert(index+1, sub_command)
                    del lines[index]
            else:
                for index in indices:
                    lines[index] = command
        # Write the file.
        file.writelines(lines)
        print(f'Wrote {FILENAME_ANSYS_SCRIPT}.')

# # Crop and resize the stress contour images.
# def crop_output_images(samples):
#     load_samples, angle_samples, length_samples, height_samples = samples
#     filenames = glob.glob(os.path.join(FOLDER_TRAIN_OUTPUTS, '*.bmp'))
#     for filename in filenames:
#         # Only crop files named as a number ("1.png").
#         try:
#             i = int(os.path.basename(filename).split('.')[0]) - 1
#         except ValueError:
#             continue
#         else:
#             with Image.open(filename) as image:
#                 # Crop out logo and some empty space.
#                 image = image.crop((0, 100, 1000, image.height-1))
#                 # Crop out all empty space and the white border around the cantilever.
#                 area = image.convert('L').getbbox()
#                 image_contour = image.crop((area[0]+1, area[1]+1, area[2]-1, area[3]-1))
#                 size_contour = (
#                     round((length_samples[i] / length.high) * OUTPUT_SIZE[0]),
#                     round((height_samples[i] / height.high) * OUTPUT_SIZE[1]),
#                     )
#                 image_contour = image_contour.resize(size_contour)
#                 # Fill the image with black.
#                 image = image.resize(OUTPUT_SIZE)
#                 image.paste((0,0,0), (0,0,image.width,image.height))
#                 # Place the contour in the black image, aligned to the top left corner.
#                 image.paste(image_contour, (0,0,size_contour[0],size_contour[1]))
#                 image.save(
#                     os.path.join(FOLDER_TRAIN_OUTPUTS, f'stress_{load_samples[i]}_{angle_samples[i]}_{length_samples[i]}_{height_samples[i]}.png')
#                     )

# # Scale and properly order stress data in text files generated by FEA and write to a CSV file.
# def convert_fea_to_spreadsheet():
#     filenames = glob.glob(os.path.join(FOLDER_TRAIN_OUTPUTS, '*.txt'))
#     assert len(filenames) == NUMBER_SAMPLES, f'Found {len(filenames)} .txt files in {FOLDER_TRAIN_OUTPUTS}, but should be {NUMBER_SAMPLES}.'
#     all_stresses = np.zeros((*OUTPUT_SIZE, NUMBER_SAMPLES))
#     for i, filename in enumerate(filenames):
#         with open(filename, 'r') as file:
#             stresses = [float(line) for line in file.readlines()]
#             # Interior nodes.
#             all_stresses[1:-1, 1:-1, i] = np.flipud(
#                 np.reshape(stresses[2*sum(NUMBER_DIVISIONS):], [_-1 for _ in NUMBER_DIVISIONS], 'F')
#                 )
#             # Corner nodes.
#             all_stresses[-1, 0, i] = stresses[0]
#             all_stresses[-1, -1, i] = stresses[1]
#             all_stresses[0, -1, i] = stresses[1+NUMBER_DIVISIONS[0]]
#             all_stresses[0, 0, i] = stresses[1+NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]]
#             # Edge nodes.
#             all_stresses[-1, 1:-1, i] = stresses[2:2+NUMBER_DIVISIONS[0]-1]
#             all_stresses[1:-1, -1, i] = stresses[2+NUMBER_DIVISIONS[0]:2+NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]-1][::-1]
#             all_stresses[0, 1:-1, i] = stresses[2+NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]:2+2*NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]-1][::-1]
#             all_stresses[1:-1, 0, i] = stresses[2+2*NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]:2+2*NUMBER_DIVISIONS[0]+2*NUMBER_DIVISIONS[1]-1]
#     # Scale all values to [0, 1].
#     all_stresses -= np.min(all_stresses)
#     all_stresses /= np.max(all_stresses)
#     # Write values to Excel file.
#     all_stresses.tofile(
#         os.path.join(FOLDER_TRAIN_OUTPUTS, FILENAME_STRESS_FEA),
#         sep=',',
#         )
#     print(f'Wrote stress data from FEA in {FOLDER_TRAIN_OUTPUTS}.')

# Get properly ordered stress data in text files generated by FEA.
def convert_fea_to_spreadsheet(samples):
    length_samples, height_samples = samples[2:4]
    filenames = glob.glob(os.path.join(FOLDER_TRAIN_OUTPUTS, '*.txt'))
    filenames = sorted(filenames)
    assert len(filenames) == NUMBER_SAMPLES, f'Found {len(filenames)} .txt files in {FOLDER_TRAIN_OUTPUTS}, but should be {NUMBER_SAMPLES}.'
    stresses = np.zeros((*OUTPUT_SIZE[::-1], len(filenames)))
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as file:
            stress = [float(line) for line in file.readlines()]
        array = np.zeros(stresses.shape[:2])
        # Interior nodes.
        array[1:-1, 1:-1] = np.flipud(
            np.reshape(stress[2*sum(NUMBER_DIVISIONS):], [_-1 for _ in NUMBER_DIVISIONS[::-1]], 'F')
            )
        # Corner nodes.
        array[-1, 0] = stress[0]
        array[-1, -1] = stress[1]
        array[0, -1] = stress[1+NUMBER_DIVISIONS[0]]
        array[0, 0] = stress[1+NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]]
        # Edge nodes.
        array[-1, 1:-1] = stress[2:2+NUMBER_DIVISIONS[0]-1]
        array[1:-1, -1] = stress[2+NUMBER_DIVISIONS[0]:2+NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]-1][::-1]
        array[0, 1:-1] = stress[2+NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]:2+2*NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]-1][::-1]
        array[1:-1, 0] = stress[2+2*NUMBER_DIVISIONS[0]+NUMBER_DIVISIONS[1]:2+2*NUMBER_DIVISIONS[0]+2*NUMBER_DIVISIONS[1]-1]
        # Resize based on the size of the cantilever.
        array_scaled = array - np.min(array)
        array_scaled = array / np.max(array_scaled)
        with Image.fromarray((array_scaled*255).astype(np.uint8), 'L') as image:
            image = image.resize((
                round((length_samples[i] / length.high) * OUTPUT_SIZE[0]),
                round((height_samples[i] / height.high) * OUTPUT_SIZE[1]),
                ))
            array_scaled = np.asarray(image, float) / 255
            array_scaled *= max(stress) - min(stress)
            array_scaled += min(stress)
        # Insert the resized array.
        array[:] = 0
        array[:array_scaled.shape[0], :array_scaled.shape[1]] = array_scaled
        stresses[:, :, i] = array
    # Scale all values to [0, 1].
    stresses -= np.min(stresses)
    stresses /= np.max(stresses)
    # Write values to spreadsheet file.
    stresses.flatten().tofile(
        os.path.join(FOLDER_TRAIN_OUTPUTS, FILENAME_STRESS_FEA),
        sep=',',
        )
    print(f'Wrote stress data from FEA in {FOLDER_TRAIN_OUTPUTS}.')

# Try to read sample values from the text file if it already exists. If not, generate the samples.
samples = read_samples()
if not samples:
    samples = generate_samples(show_histogram=False)
load_components = write_samples(samples)
write_ansys_script(samples, load_components)
generate_input_images(samples, FOLDER_TRAIN_INPUTS)

convert_fea_to_spreadsheet(samples)
# # Crop and resize stress contour images generated by FEA. This only needs to run the first time the images are added to the folder.
# crop_output_images(samples)