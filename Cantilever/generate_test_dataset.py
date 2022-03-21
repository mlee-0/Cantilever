'''
Write files used for testing.
'''

from setup import *


NUMBER_SAMPLES = 10

if __name__ == '__main__':
    # Try to read sample values from the text file if it already exists. If not, generate the samples.
    samples = read_samples(FILENAME_SAMPLES_TEST)
    if samples:
        overwrite_existing_samples = input(f"Overwrite {FILENAME_SAMPLES_TEST}? [y/n] ") == "y"
    else:
        overwrite_existing_samples = False
    if not samples or overwrite_existing_samples:
        samples = generate_samples(NUMBER_SAMPLES)
    write_samples(samples, FILENAME_SAMPLES_TEST)
    write_ansys_script(samples, 'ansys_script_test.lgw')