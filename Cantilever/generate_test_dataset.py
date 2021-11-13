from main import generate_input_images, FOLDER_TEST_INPUTS


# Generate new data not part of the training dataset.
test_loads = [25000, 50000, 75000, 100000]
test_angles = [5, 30, 90, 175]
test_lengths = [3] * 4
test_heights = [1.5] * 4
test_filenames = generate_input_images((test_loads, test_angles, test_lengths, test_heights), FOLDER_TEST_INPUTS)