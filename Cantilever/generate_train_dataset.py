'''
Write files used for training.
'''

from setup import *


NUMBER_SAMPLES = 800

if __name__ == '__main__':
    # Try to read sample values from the text file if it already exists. If not, generate the samples.
    samples = read_samples(FILENAME_SAMPLES_TRAIN)
    if not samples:
        samples = generate_samples(NUMBER_SAMPLES)
    write_samples(samples, FILENAME_SAMPLES_TRAIN)
    write_ansys_script(samples, 'ansys_script_train.lgw')