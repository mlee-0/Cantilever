'''
Write files used for training and validation.
'''

from setup import *


NUMBER_SAMPLES = 100_000

if __name__ == '__main__':
    # Try to read sample values from the text file if it already exists. If not, generate the samples.
    samples = read_samples(FILENAME_SAMPLES_TRAIN)
    if samples:
        overwrite_existing_samples = input(f"Overwrite {FILENAME_SAMPLES_TRAIN}? [y/n] ") == "y"
    else:
        overwrite_existing_samples = False
    if not samples or overwrite_existing_samples:
        samples = generate_samples(NUMBER_SAMPLES)
    write_samples(samples, FILENAME_SAMPLES_TRAIN)
    write_ansys_script(samples, 'ansys_script_train.lgw')