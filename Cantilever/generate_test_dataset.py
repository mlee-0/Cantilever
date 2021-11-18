from main import FOLDER_TEST_INPUTS, FOLDER_TEST_OUTPUTS, generate_samples, generate_input_images, calculate_load_components, write_ansys_script, write_fea_spreadsheet
from generate_train_dataset import write_samples, read_samples

NUMBER_SAMPLES = 8
FILENAME_SAMPLES = 'samples_test.txt'

if __name__ == '__main__':
    # Try to read sample values from the text file if it already exists. If not, generate the samples.
    samples = read_samples(FILENAME_SAMPLES)
    if not samples:
        samples = generate_samples(NUMBER_SAMPLES, show_histogram=False)
    load_components = calculate_load_components(*samples[0:2])
    write_samples(samples, load_components, FILENAME_SAMPLES)
    write_ansys_script(samples, load_components, 'ansys_script_test.lgw')
    generate_input_images(samples, FOLDER_TEST_INPUTS)
    write_fea_spreadsheet(samples, FOLDER_TEST_OUTPUTS, 'stress.csv')
    # # Generate new data not part of the training dataset.
    # # test_loads = [90000] * 4 + [30000, 50000, 70000, 90000]
    # # test_angles = [5, 10, 15, 30] + [270] * 4
    # # test_lengths = [3] * 8
    # # test_heights = [1.5] * 8
    # # samples = (test_loads, test_angles, test_lengths, test_heights)
    # samples = generate_samples(5, show_histogram=False)
    # load_components = calculate_load_components(*samples[0:2])
    # write_ansys_script(samples, load_components, 'ansys_script_test.lgw')
    # generate_input_images(samples, FOLDER_TEST_INPUTS)