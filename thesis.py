"""
Functions for generating figures.
"""


import matplotlib.pyplot as plt

from helpers import *


def plot_page_progress(current: int, previous: int, goal_1: int=120, goal_2: int=150):
    """Figure showing a bar graph of the current number of pages compared to the desired number of pages."""

    y = 2.5
    plt.figure()
    bar_goal_2 = plt.barh(y, goal_2, color=[0.95]*3)
    bar_goal_1 = plt.barh(y, goal_1, color=[0.90]*3, )
    bar_current = plt.barh(y, current, color='#00BF60', label='This week')
    bar_previous = plt.barh(y, previous, color=[0.00]*3)
    plt.bar_label(bar_goal_2)
    plt.bar_label(bar_goal_1)
    plt.bar_label(bar_current)
    plt.bar_label(bar_previous)
    plt.legend()
    plt.ylim([0, 5])
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_label_histogram():
    """Figure of histogram of all labels in dataset."""

    labels = read_pickle('Labels 2D/labels.pickle')
    print(labels.mean())

    plt.figure()
    plt.hist(labels[labels != -1], bins=100)
    plt.xlabel('Stress [Pa]')
    plt.xticks([0, 20_000, 40_000, labels.max()])
    plt.yticks([])
    plt.show()

def plot_input():
    """Figure of both channels of an input."""

    samples = read_samples('samples.csv')
    inputs = generate_input_images(samples, is_3d=False)
    i = 829 #5200

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(inputs[i, 0, ...], cmap='gray', vmin=0, vmax=2)
    plt.title(f"Length = {samples['Length'][i]} m, Height = {samples['Height'][i]} m")
    plt.colorbar(fraction=0.0225, ticks=[0, 1, 2])
    plt.subplot(1, 2, 2)
    plt.imshow(inputs[i, 1, ...], cmap='gray', vmin=0, vmax=2)
    plt.title(f"Angle = {int(samples['Angle XY'][i])}°")
    plt.colorbar(fraction=0.0225, ticks=[0, 1, 2])
    plt.show()

def plot_label():
    """Figure of a typical stress distribution with corresponding color scale."""

    labels = read_pickle('Labels 2D/labels.pickle')
    label = labels[5200, 0, ...]
    # label = labels[829, 0, ...]

    plt.figure()
    plt.imshow(label, cmap='Spectral_r', vmin=label.max(), vmax=0)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.02, ticks=np.linspace(0, label.max(), 4), label='Stress [Pa]')
    plt.show()

def plot_stratified_sampling_histogram():
    """Figure of a histogram of maximum stresses in the dataset. Hard-coded to recreate the figure made on 2022-03-25."""

    # labels = read_pickle('Labels 2D/labels.pickle')
    # labels = np.reshape(labels, (labels.shape[0], -1))
    # maxima = np.max(labels, axis=1)

    data = np.array([130, 462, 738, 477, 291, 184, 99, 65, 23, 6])
    edges = np.array([0, 2744, 5488, 8232, 10976, 13720, 16464, 19208, 21952, 24696, 27440])
    plt.figure()
    plt.bar(edges[:-1], data, width=np.diff(edges), align='edge')
    plt.text(np.mean(edges[-2:]), data[-1]*2, str(data[-1]), fontdict={'fontweight': 'bold'})
    plt.xticks(edges, rotation=90)
    plt.xlabel('Maximum Stress [Pa]')
    plt.tight_layout()
    plt.show()

def plot_stratified_sampling_metrics():
    """Figure comparing evaluation metrics for models trained both with and without stratified sampling."""

    # First column is without, second column is with stratified sampling.
    me = np.array([[2.03, 1.98], [-8.42, 3.36], [2.52, 11.97], [-126.76, 89.86], [-552.78, -134.31]])[::-1, :]
    mae = np.array([[56.11, 55.93], [76.03, 73.24], [188.27, 156.97], [510.20, 371.13], [1051.73, 772.29]])[::-1, :]
    mse = np.array([[13285, 10075], [34861, 17283], [208906, 81686], [1267646, 547539], [4147007, 2317887]])[::-1, :]
    mre = np.array([[4.90, 5.17], [7.00, 5.90], [12.30, 10.62], [23.97, 21.38], [43.19, 39.19]])[::-1, :]
    dataset_sizes = np.array([2000, 1000, 500, 200, 100])[::-1]

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.grid()
    plt.plot(me, '*-', label=['Normal', 'Sampling'])
    plt.title('Mean Error')
    plt.xlabel('Dataset Size')
    plt.xticks(range(5), labels=dataset_sizes)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.grid()
    plt.semilogy(mae, '*-', label=['Normal', 'Sampling'])
    plt.title('MAE')
    plt.xlabel('Dataset Size')
    plt.xticks(range(5), labels=dataset_sizes)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.grid()
    plt.semilogy(mse, '*-', label=['Normal', 'Sampling'])
    plt.title('MSE')
    plt.xlabel('Dataset Size')
    plt.xticks(range(5), labels=dataset_sizes)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.grid()
    plt.semilogy(mre, '*-', label=['Normal', 'Sampling'])
    plt.title('MRE')
    plt.xlabel('Dataset Size')
    plt.xticks(range(5), labels=dataset_sizes)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_page_progress(current=14, previous=2, goal_1=120, goal_2=150)

    # plot_stratified_sampling_metrics()