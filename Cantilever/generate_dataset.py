'''
Write files used for training/validation and testing.
'''

from setup import *


DATASET = 'train'
NUMBER_SAMPLES = 10
START_SAMPLE_NUMBER = 1
WRITE_MODE = 'w'

if __name__ == '__main__':
    DATASET_TYPES = ['train', 'test']
    filename_sample = dict(zip(DATASET_TYPES, [FILENAME_SAMPLES_TRAIN, FILENAME_SAMPLES_TEST]))[DATASET]
    filename_ansys = dict(zip(DATASET_TYPES, ['ansys_script_train.lgw', 'ansys_script_test.lgw']))[DATASET]

    assert DATASET in DATASET_TYPES
    assert not (START_SAMPLE_NUMBER > 1 and WRITE_MODE == 'w'), f"The starting sample number {START_SAMPLE_NUMBER} must be 1 when generating a new dataset."

    # Try to read sample values from the text file if it already exists. If not, generate the samples.
    samples = read_samples(filename_sample)
    if samples:
        overwrite_existing_samples = input(f"Overwrite {filename_sample}? [y/n] ") == "y"
    else:
        overwrite_existing_samples = False
    if not samples or overwrite_existing_samples:
        samples = generate_samples(NUMBER_SAMPLES, start=START_SAMPLE_NUMBER)

    write_samples(samples, filename_sample, mode=WRITE_MODE)
    write_ansys_script(samples, filename_ansys)