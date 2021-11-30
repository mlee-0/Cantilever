from main import FOLDER_TEST_INPUTS, FOLDER_TEST_OUTPUTS, FILENAME_SAMPLES_TEST, generate_samples, write_samples, read_samples, generate_input_images, write_ansys_script
from generate_train_dataset import write_samples, read_samples


NUMBER_SAMPLES = 8

if __name__ == '__main__':
    # Try to read sample values from the text file if it already exists. If not, generate the samples.
    samples = read_samples(FILENAME_SAMPLES_TEST)
    if not samples:
        samples = generate_samples(NUMBER_SAMPLES, show_histogram=False)
    write_samples(samples, FILENAME_SAMPLES_TEST)
    write_ansys_script(samples, 'ansys_script_test.lgw')
    generate_input_images(samples, FOLDER_TEST_INPUTS)