from main import FOLDER_TEST_INPUTS, generate_samples, generate_input_images, calculate_load_components, write_ansys_script


if __name__ == '__main__':
    # Generate new data not part of the training dataset.
    test_loads = [90000] * 4 + [30000, 50000, 70000, 90000]
    test_angles = [5, 10, 15, 30] + [270] * 4
    test_lengths = [3] * 8
    test_heights = [1.5] * 8
    samples = (test_loads, test_angles, test_lengths, test_heights)
    load_components = calculate_load_components(*samples[0:2])
    write_ansys_script(samples, load_components, 'ansys_script_test.lgw')
    generate_input_images(samples, FOLDER_TEST_INPUTS)