"""Functions for generating figures."""


import random

from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image

from preprocessing import *
from datasets import CantileverDataset, generate_simulation_parameters


def plot_page_progress(current: int, previous: int, goal_1: int=120, goal_2: int=150):
    """Bar graph of the current number of pages compared to the desired number of pages."""

    y = 1
    plt.figure(figsize=(6, 2))
    plt.subplots_adjust(left=0.01, right=0.99)
    # bar_goal_2 = plt.barh(y, goal_2, color=[0.95]*3)
    bar_goal_1 = plt.barh(y, goal_1, color=[0.90]*3)
    bar_current = plt.barh(y, current, color='#00BF60', label='This week')
    bar_previous = plt.barh(y, previous, color=[0.00]*3)
    # plt.bar_label(bar_goal_2)
    plt.bar_label(bar_goal_1)
    plt.bar_label(bar_current, fontweight='bold')
    plt.bar_label(bar_previous)
    plt.legend()
    plt.ylim([0, 2])
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_page_history():
    """Cumulative pages by day and pages added by day."""

    from datetime import datetime
    from matplotlib.dates import DateFormatter
    pages = {
        '2023-01-26': 0,
        '2023-02-19': 0,
        '2023-02-20': 2,
        '2023-02-21': 3,
        '2023-02-22': 3,
        '2023-02-23': 8,
        '2023-02-24': 9,
        '2023-02-25': 13,
        '2023-02-26': 18,
        '2023-02-28': 20,
        '2023-03-02': 24,
        '2023-03-04': 27,
        '2023-03-05': 31,
        '2023-03-06': 31,
        '2023-03-07': 32,
        '2023-03-08': 34,
        '2023-03-09': 38,
        '2023-03-11': 42,
        '2023-03-12': 46,
        '2023-03-13': 51,
        '2023-03-15': 56,
        '2023-03-16': 60,
        '2023-03-19': 65,
        '2023-03-20': 66,
        '2023-03-21': 75,
        '2023-03-22': 79,
        '2023-03-23': 85,
        '2023-03-24': 85,
        '2023-03-25': 87,
        '2023-03-26': 90,
        '2023-03-27': 91,
        '2023-03-28': 93,
        '2023-03-29': 94,
        '2023-03-30': 98,
        '2023-03-31': 100,
        '2023-04-01': 105,
        '2023-04-02': 109,
        '2023-04-03': 111,
        '2023-04-05': 113,
        '2023-04-06': 114,
        '2023-04-07': 116,
        '2023-04-08': 117,
        '2023-04-09': 120,
        '2023-04-10': 121,
        '2023-04-19': 122,
        '2023-04-26': 121,
        '2023-04-27': 123,
    }
    dates = [datetime.strptime(_, '%Y-%m-%d') for _ in pages.keys()]

    plt.figure()
    plt.grid(axis='y')
    plt.bar(dates, np.diff([0, *pages.values()]), width=1.0, color=[0/255, 191/255, 96/255], alpha=1/3, label='Daily')
    plt.axhline(y=120, color=[0.25]*3, linewidth=2)
    plt.step(dates, pages.values(), where='mid', color=[0/255, 191/255, 96/255], label='Cumulative')
    plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d'))
    plt.ylim([0, 130])
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def plot_page_distribution():
    """Pie chart of pages by chapter."""

    page_counts = np.diff([1, 12, 40, 92, 113, 117])
    goal = 120

    plt.figure(figsize=(4, 4))
    plt.pie(
        [_/goal for _ in page_counts],
        normalize=False,
        labels=[f'Chapter {i+1}\n{count}' for i, count in enumerate(page_counts)],
        pctdistance=0.8,
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.25, 'edgecolor': [1.0]*3},
    )
    plt.show()

def plot_data_imbalance_classification():
    """Visualization of data imbalance in classification."""

    plt.figure(figsize=(3, 3))
    plt.bar([1, 2], [90, 10])
    plt.xticks(ticks=[1, 2], labels=['Not Cancer', 'Cancer'])
    plt.yticks([])
    plt.title('Dataset')
    plt.show()

def plot_function_approximation():
    """Visualization of function approximation."""

    f = lambda x: x ** 2

    x_data = np.random.rand(15) * 3
    y_data = f(x_data) + np.random.randn(x_data.size)

    x_fit = np.linspace(0, 3, 100)
    y_fit = f(x_fit)

    plt.scatter(x_data, y_data)
    plt.plot(x_fit, y_fit, '--')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_label_histogram():
    """Figure of histogram of all labels in dataset."""

    labels = read_pickle('Labels 2D/labels.pickle')
    print(f"Mean {labels.mean()}, median {np.median(labels)}")

    plt.figure()
    plt.hist(labels[labels != -1], bins=100)
    plt.xlabel('Stress [Pa]')
    plt.xticks([0, 20_000, 40_000, labels.max()])
    plt.show()

def plot_inputs():
    """Both channels of an input image."""

    parameters = generate_simulation_parameters()
    inputs = make_inputs(parameters)
    i = 5200  #829

    plt.figure(figsize=(4, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(inputs[i, 0, ...], cmap='gray', vmin=0, vmax=2)
    plt.title(f"Length = {parameters[i][1]} m, Height = {parameters[i][2]} m")
    plt.colorbar(fraction=0.023, ticks=[0, 1, 2])
    plt.subplot(2, 1, 2)
    plt.imshow(inputs[i, 1, ...], cmap='gray', vmin=0, vmax=2)
    plt.title(f"Angle = {int(parameters[i][0])}°")
    plt.colorbar(fraction=0.023, ticks=[0, 1, 2])
    plt.show()

def plot_labels():
    """Examples of stress distributions with corresponding color scale."""

    labels = read_pickle('Labels 2D/labels.pickle')
    # label = labels[5200, 0, ...]

    plt.figure(figsize=(6, 3))
    for i, index in enumerate(random.sample(list(range(labels.shape[0])), k=6)):
        label = labels[index, 0, ...]
        plt.subplot(3, 2, i+1)
        plt.imshow(label, cmap='Spectral_r', vmin=label.max(), vmax=0)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(fraction=0.02, ticks=[label.min(), label.max()], label='Stress [Pa]')
    plt.tight_layout()
    plt.show()

def plot_inputs_labels():
    """Many input images with corresponding labels."""

    dataset = CantileverDataset()

    for i, index in enumerate([1000, 3000, 5000, 7000]):
        input_data, label_data = dataset[index]

        plt.subplot(5, 3, i*3 + 1)
        plt.imshow(input_data[0, ...], cmap='gray')
        plt.subplot(5, 3, i*3 + 2)
        plt.imshow(input_data[1, ...], cmap='gray')
        plt.subplot(5, 3, i*3 + 3)
        plt.imshow(label_data[0, ...], cmap='Spectral_r')

    plt.show()

def plot_evaluation_metrics():
    """Evaluation metric functions."""

    error = np.linspace(-5, 5, 100)

    plt.figure()
    plt.grid()
    plt.plot(error, error ** 2, label='MSE')
    plt.plot(error, np.abs(error), '-', label='MAE')
    plt.plot(error, np.sqrt(error ** 2), '--', label='RMSE')
    plt.xlabel('Error')
    plt.legend()
    plt.show()

def plot_hyperparameter_tuning_random_search(metric_name: Literal['mae', 'mse', 'rmse', 'mre']):
    """Random search results"""

    results = {
        (-3.1273, 59, 51): (0.061, 0.022, 0.150, 15.010),
        (-0.2464, 127, 7): (0.786, 7.430, 2.726, 34.358),
        (-1.34, 42, 21): (0.109, 0.071, 0.267, 22.648),
        (-2.0067, 128, 9): (0.105, 0.082, 0.287, 58.697),
        (-4.2199, 92, 39): (0.127, 0.100, 0.317, 51.032),
        (-4.22, 60, 18): (0.167, 0.200, 0.447, 115.045),
        (-4.7096, 80, 4): (0.793, 6.153, 2.481, 1139.511),
        (-0.6691, 15, 25): (0.786, 7.430, 2.726, 34.358),
        (-1.9944, 62, 60): (0.098, 0.122, 0.350, 30.846),
        (-1.4596, 62, 14): (0.111, 0.105, 0.324, 38.328),
        (-4.8971, 47, 50): (0.191, 0.270, 0.520, 139.989),
        (-0.1505, 62, 58): (0.786, 7.430, 2.726, 34.358),
        (-0.8378, 51, 9): (0.147, 0.128, 0.357, 35.688),
        (-3.9383, 108, 26): (0.123, 0.118, 0.343, 40.332),
        (-4.0909, 55, 53): (0.081, 0.085, 0.291, 19.147),
        (-4.083, 116, 2): (0.598, 4.747, 2.179, 742.315),
        (-3.4788, 64, 20): (0.084, 0.056, 0.237, 25.062),
        (-2.3762, 121, 28): (0.080, 0.054, 0.233, 19.360),
        (-2.8403, 3, 47): (0.049, 0.026, 0.160, 10.869),
        (-3.5439, 101, 60): (0.096, 0.120, 0.346, 32.817),
    }

    hyperparameter_names = ['Learning Rate Exponent', 'Batch Size', 'Model Size']

    hyperparameters = list(zip(*results.keys()))
    metrics = list(zip(*results.values()))
    if metric_name == 'mae':
        metrics = metrics[0]
    elif metric_name == 'mse':
        metrics = metrics[1]
    elif metric_name == 'rmse':
        metrics = metrics[2]
    elif metric_name == 'mre':
        metrics = metrics[3]
    metrics = np.array(metrics)

    plt.figure()
    for j, hyperparameter in enumerate(hyperparameters):
        plt.subplot(1, len(hyperparameter_names), j+1)
        plt.scatter(hyperparameter, metrics)
        plt.xlabel(hyperparameter_names[j])
    plt.tight_layout()

    colormap = cm.get_cmap('Greens_r')
    plt.figure()
    for row, hyperparameter_1 in enumerate(hyperparameters):
        for column, hyperparameter_2 in enumerate(hyperparameters):
            if hyperparameter_1 is not hyperparameter_2 and column >= row:
                plt.subplot(len(hyperparameters), len(hyperparameters), row*len(hyperparameters)+(column+1))
                plt.grid()
                plt.scatter(hyperparameter_1, hyperparameter_2, color=colormap(metrics / metrics.max()))
                plt.xlabel(hyperparameter_names[row])
                plt.ylabel(hyperparameter_names[column])
    plt.tight_layout()

    plt.show()

def plot_hyperparameter_tuning_grid_search(metric_name: Literal['mae', 'mse', 'rmse', 'mre']):
    """Grid search results."""

    # Best results, not necessarily at end of 10 epochs.
    results = {
        (4, 16): (0.06913479, 0.05132358, 0.22654708, 15.065224468708038),
        (4, 32): (0.05792376, 0.029064063, 0.17048186, 14.003099501132965),
        (4, 64): (0.050798096, 0.021032125, 0.14502457, 12.044135481119156),
        (8, 16): (0.06938156, 0.032970194, 0.18157697, 15.338702499866486),
        (8, 32): (0.06445467, 0.033481278, 0.1829789, 17.66492873430252),
        (8, 64): (0.07737578, 0.040889885, 0.20221248, 19.335877895355225),
        (16, 16): (0.06512222, 0.033447757, 0.18288729, 15.748365223407745),
        (16, 32): (0.049868856, 0.01937678, 0.13920051, 11.327722668647766),
        (16, 64): (0.06872461, 0.037771214, 0.19434817, 15.472258627414703),
        (32, 16): (0.06278347, 0.03689035, 0.1920686, 17.731794714927673),
        (32, 32): (0.07862071, 0.051436126, 0.22679535, 16.146492958068848),
        (32, 64): (0.05569741, 0.024537234, 0.15664366, 13.449710607528687),
    }

    hyperparameter_names = ['Batch Size', 'Model Size']

    hyperparameters = [np.array(_) for _ in zip(*results.keys())]
    metrics = list(zip(*results.values()))
    if metric_name == 'mae':
        metrics = metrics[0]
    elif metric_name == 'mse':
        metrics = metrics[1]
    elif metric_name == 'rmse':
        metrics = metrics[2]
    elif metric_name == 'mre':
        metrics = metrics[3]
    metrics = np.reshape(metrics, (4, 3))

    plt.imshow(metrics, cmap='Greens_r', vmin=metrics.min(), vmax=metrics.max())
    plt.xticks(ticks=range(metrics.shape[1]), labels=np.unique(hyperparameters[1]))
    plt.yticks(ticks=range(metrics.shape[0]), labels=np.unique(hyperparameters[0]))
    plt.xlabel(hyperparameter_names[1])
    plt.ylabel(hyperparameter_names[0])
    # for axis_i in range(metrics.ndim):
    #     axis_j, axis_k = [_ for _ in range(metrics.ndim) if _ != axis_i]

    #     plt.figure()
    #     for subplot in range(metrics.shape[axis_i]):
    #         plt.subplot(1, metrics.shape[axis_i], subplot+1)
    #         plt.imshow(metrics.take(subplot, axis=axis_i), cmap='Greens_r', vmin=metrics.min(), vmax=metrics.max())
    #         plt.xticks(ticks=range(metrics.shape[axis_k]), labels=np.unique(hyperparameters[axis_k]))
    #         plt.yticks(ticks=range(metrics.shape[axis_j]), labels=np.unique(hyperparameters[axis_j]))
    #         plt.xlabel(hyperparameter_names[axis_k])
    #         plt.ylabel(hyperparameter_names[axis_j])
    #         plt.title(f"{hyperparameter_names[axis_i]} = {np.unique(hyperparameters[axis_i])[subplot]}")
    #         # colorbar = plt.colorbar(ticks=[metrics.min(), metrics.max()], fraction=0.01)
    #         # colorbar.set_ticklabels(['Best', 'Worst'])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    plt.show()

def plot_stratified_sampling_histogram():
    """Histogram of maximum stresses in the dataset. Hard-coded to recreate the figure made on 2022-03-25."""

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

    # First column is baseline, second column is stratified sampling.
    me = np.array([[2.03, 1.98], [-8.42, 3.36], [2.52, 11.97], [-126.76, 89.86], [-552.78, -134.31]])[::-1, :]
    mae = np.array([[56.11, 55.93], [76.03, 73.24], [188.27, 156.97], [510.20, 371.13], [1051.73, 772.29]])[::-1, :]
    mse = np.array([[13285, 10075], [34861, 17283], [208906, 81686], [1267646, 547539], [4147007, 2317887]])[::-1, :]
    mre = np.array([[4.90, 5.17], [7.00, 5.90], [12.30, 10.62], [23.97, 21.38], [43.19, 39.19]])[::-1, :]
    dataset_sizes = np.array([2000, 1000, 500, 200, 100])[::-1]

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.grid()
    plt.plot(me[:, 0], '.-', color=[0.0]*3, label='Baseline')
    plt.plot(me[:, 1], '.-', label='SS')
    plt.title('Mean Error')
    plt.xlabel('Dataset Size')
    plt.xticks(range(5), labels=dataset_sizes)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.grid()
    plt.semilogy(mae[:, 0], '.-', color=[0.0]*3, label='Baseline')
    plt.semilogy(mae[:, 1], '.-', label='SS')
    plt.title('MAE')
    plt.xlabel('Dataset Size')
    plt.xticks(range(5), labels=dataset_sizes)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.grid()
    plt.semilogy(mse[:, 0], '.-', color=[0.0]*3, label='Baseline')
    plt.semilogy(mse[:, 1], '.-', label='SS')
    plt.title('MSE')
    plt.xlabel('Dataset Size')
    plt.xticks(range(5), labels=dataset_sizes)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.grid()
    plt.semilogy(mre[:, 0], '.-', color=[0.0]*3, label='Baseline')
    plt.semilogy(mre[:, 1], '.-', label='SS')
    plt.title('MRE')
    plt.xlabel('Dataset Size')
    plt.xticks(range(5), labels=dataset_sizes)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_stratified_sampling_predictions():
    import glob

    folders = glob.glob('Stratified Sampling Predictions/*')

    for folder in folders:
        plt.figure(figsize=(6,3))

        files = glob.glob(f"{folder}/*.png")
        for i, file in enumerate(sorted(files)):
            with Image.open(file, 'r') as image:
                array = np.asarray(image)
                true = array[:array.shape[0]//2, :]
                prediction = array[array.shape[0]//2:, :]
            
            # Show prediction.
            plt.subplot(3, 4, 5 + i)
            plt.imshow(prediction)
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel('Baseline')
            elif i == 4:
                plt.ylabel('SS')

            # Show label.
            if 's' in file:
                plt.subplot(3, 4, 1 + i%4)
                plt.imshow(true)
                plt.xticks([])
                plt.yticks([])
                if i == 0:
                    plt.ylabel('True')
    
            plt.suptitle(f"{os.path.basename(folder)} Data", fontweight='bold')
        plt.tight_layout()

    plt.show()

def plot_label_transformation_average_histogram():
    """Raw histogram and average sample histogram (both normalized)."""

    data = read_pickle('Labels 2D/labels.pickle')
    data /= data.max()

    transformed = data.copy()
    transformed /= transformed.max(axis=tuple(range(1, data.ndim)), keepdims=True)

    plt.hist(data.flatten(), bins=100, alpha=0.5, label='Original')
    plt.hist(transformed.flatten(), bins=100, alpha=0.5, label='Average Sample')
    plt.legend()
    plt.show()

def plot_lt_powers_histogram():
    """Histograms of different exponentiation transformations."""

    data = read_pickle('Labels 2D/labels.pickle')
    data = data[data > 0].flatten()
    data /= data.max()

    for i, power in enumerate(np.arange(1.25, 4.01, 0.25)):
        transformed = data ** (1 / power)
        plt.subplot(3, 4, i+1)
        plt.hist(data, bins=100, color=[0.5]*3, alpha=0.5, label=f'Original')
        plt.hist(transformed, bins=100, alpha=0.5, label=f'1/{power}')
        plt.xticks([])
        plt.yticks([])
        plt.legend()
    
    plt.show()

def plot_lt_powers_functions():
    """Graphs of different exponentiation transformations."""

    x = np.linspace(0, 1, 100)
    plt.plot(x, x, '--', color=[0.5]*3, label='No transformation')

    for power in np.arange(1.25, 4.01, 0.25):
        plt.plot(x, x ** power, label=f'1/{power}')
    
    plt.legend()
    plt.show()

def plot_lt_logarithms_histogram():
    """Histograms of different logarithm transformations."""

    data = read_pickle('Labels 2D/labels.pickle')
    data = data[data > 0].flatten()
    data /= data.max()

    size = 1
    x1s = [1.0, 0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-10]

    for i, x1 in enumerate(x1s):
        transformed = data.copy()
        transformed -= transformed.min()
        transformed /= transformed.max()
        transformed *= size
        transformed += x1
        transformed = np.log(transformed)
        transformed -= transformed.min()
        transformed /= transformed.max()
        plt.subplot(2, 4, i+1)
        plt.hist(data, bins=100, color=[0.5]*3, alpha=0.5, label=f'Original')
        plt.hist(transformed, bins=100, alpha=0.5, label=f'({x1}, 1+{x1})')
        plt.xticks([])
        plt.yticks([])
        plt.legend()
    
    plt.show()

def plot_lt_logarithms_domain():
    """Plot showing how logarithmic transformations are obtained by selecting a portion of a natural log."""
    
    x = np.linspace(1e-5, 2, 1000)
    y = np.log(x)

    x1 = 0.01
    x2 = x1 + 1
    mask = (x > x1) & (x < x2)

    plt.figure(figsize=(4, 3))
    plt.grid()
    plt.plot(x, y, color=[0.5]*3, linewidth=1)
    plt.plot(x[mask], y[mask], linewidth=3)
    plt.axvline(x1, linestyle=':', color=[0.5]*3)
    plt.axvline(x2, linestyle=':', color=[0.5]*3)
    plt.xticks([x1, x2])
    plt.yticks([0])
    plt.show()

def plot_lt_logarithms_functions():
    """Graphs of different logarithm transformations."""

    size = 1
    x1s = [1.0, 0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-10]
    plt.plot(np.linspace(0, 1, 1000), '--', color=[0.5]*3, label='No transformation')
    for i, x1 in enumerate(x1s):
        x = np.linspace(x1, x1+size, 1000)
        y = np.log(x)
        y -= y.min()
        y /= y.max()
        plt.plot(y, '-', label=f'[{x1}, 1+{x1}]')
        plt.legend()

    plt.xticks([])
    plt.legend()
    plt.show()

def plot_lt_metrics(transform: Literal['exp', 'log']):
    """Evaluation metrics for models trained with different transformations. Specify which transformation's results to show."""

    results_exponentiation = np.array([
        [7.9987, 24.679495, 10782.223, 103.83748, 0.054158606, 0.05192430483639175, 0.22786905, 4.296436160802841, -136.7292, 555.6768, 519277.44, 720.6091, 0.068669535, 0.00793018182422635, 0.08905157, 8.499367535114288],
        [-11.426707, 28.28825, 19424.354, 139.37128, 0.062077958, 0.09354256175271285, 0.3058473, 4.718296602368355, -886.58777, 930.30676, 1676946.4, 1294.9696, 0.11496563, 0.025609604236323604, 0.16003, 11.780910193920135],
        [3.702548, 37.078655, 36211.992, 190.2945, 0.08136831, 0.17438737711186747, 0.41759717, 4.9101874232292175, -293.04633, 682.8758, 835414.75, 914.01025, 0.084388554, 0.012758094975271481, 0.11295174, 9.729297459125519],
        [0.63365155, 19.331297, 5920.115, 76.94228, 0.04242211, 0.028509701593298933, 0.16884816, 4.157403111457825, -143.24911, 433.18835, 309909.2, 556.6949, 0.053532634, 0.004732799903083129, 0.068795346, 7.103805243968964],
        [-16.685993, 24.093767, 8496.485, 92.176384, 0.052873246, 0.04091681536169548, 0.20227906, 4.978133365511894, -236.27664, 443.35367, 346819.16, 588.9135, 0.054788847, 0.005296473080803301, 0.07277687, 6.800913065671921],
        [20.782646, 30.892614, 13221.378, 114.98425, 0.06779317, 0.06367064229407914, 0.25233042, 6.047312170267105, 110.81973, 467.2452, 342068.3, 584.8661, 0.057741318, 0.0052239202371107815, 0.0722767, 6.967458873987198],
        [-24.597153, 30.456278, 18258.375, 135.12355, 0.066835634, 0.08792746973650914, 0.29652566, 5.7304747402668, -611.0491, 660.9262, 779511.06, 882.89923, 0.081676066, 0.011904357888880682, 0.10910709, 8.970752358436584],
        [-0.5516696, 17.674067, 4542.057, 67.39478, 0.03878535, 0.02187333707098249, 0.14789636, 11.698953807353973, -43.0056, 379.8246, 254243.0, 504.22516, 0.046938036, 0.003882689814607591, 0.062311236, 5.8918967843055725],
        [-18.129675, 23.837383, 7206.758, 84.892624, 0.05231061, 0.03470582100448338, 0.18629499, 11.843579262495041, -255.76085, 401.44812, 277615.3, 526.89215, 0.049610235, 0.004239621724975135, 0.06511238, 5.811024084687233],
        [-1.7691576, 20.721878, 5505.4775, 74.198906, 0.045473717, 0.026512927303431896, 0.16282791, 61.136698722839355, -56.847607, 331.50977, 205032.03, 452.80463, 0.04096738, 0.003131161052232236, 0.055956777, 4.9522023648023605],
        [0.032291014, 18.51287, 5554.6357, 74.52943, 0.04062609, 0.026749649491570528, 0.1635532, 98.50154519081116, -95.56946, 367.6348, 281610.88, 530.6702, 0.045431644, 0.004300640238060562, 0.065579265, 5.399021133780479],
        [4.60984, 20.740469, 5251.4233, 72.466705, 0.0455145, 0.02528945575867015, 0.1590266, 397.6400136947632, 7.3831654, 272.31485, 138784.64, 372.53812, 0.03365218, 0.0021194593777553853, 0.04603759, 3.94560769200325],
    ])
    results_logarithm = np.array([
        [-46.61707, 81.62808, 104194.875, 322.7923, 0.17913103, 0.5017747591591175, 0.7083606, 8.379248529672623, -1134.9989, 1603.0781, 4845543.0, 2201.2595, 0.20172721, 0.0767294472958232, 0.2770008, 21.441812813282013],
        [-30.642654, 43.472458, 48549.98, 220.3406, 0.09539936, 0.2338038029205155, 0.48353264, 4.73240464925766, -1127.0745, 1166.425, 2524188.0, 1588.7693, 0.14677992, 0.03997065965790612, 0.19992663, 15.237011015415192],
        [-9.872236, 35.560734, 36450.848, 190.92105, 0.07803724, 0.17553754783118777, 0.41897202, 6.669990718364716, -281.2654, 816.2812, 1680148.4, 1296.2053, 0.10271872, 0.026605244487300082, 0.16311114, 9.931723773479462],
        [-2.4943182, 27.800255, 17629.07, 132.7745, 0.06100704, 0.08489689464517118, 0.29137072, 8.853719383478165, -49.512188, 684.5157, 1080435.1, 1039.4398, 0.08613769, 0.017108751275191174, 0.13080043, 9.44911539554596],
        [33.017124, 41.463768, 46593.28, 215.85477, 0.09099132, 0.22438079940320618, 0.4736885, 15.29935747385025, 422.57376, 792.5354, 1651041.6, 1284.9286, 0.09973061, 0.026144337455812034, 0.1616921, 9.509637206792831],
        [-22.574715, 31.452307, 25291.91, 159.0343, 0.06902139, 0.12179908491191409, 0.34899724, 23.956964910030365, -459.74298, 749.3136, 1341869.0, 1158.3907, 0.094291694, 0.021248571463177397, 0.14576891, 9.128406643867493],
        [-8.97186, 19.09661, 7464.204, 86.39562, 0.041907087, 0.035945618394148664, 0.18959329, 73.60569834709167, -172.23703, 429.2404, 403399.3, 635.13727, 0.054014504, 0.006387850915292686, 0.07992403, 6.19758740067482],
        [11.316159, 22.93864, 17228.13, 131.25598, 0.050338347, 0.08296609953791545, 0.28803837, 222.18806743621826, 13.44763, 405.512, 490360.97, 700.2578, 0.051028583, 0.007764893657456334, 0.088118635, 5.601198971271515],
        [-1.1945096, 19.379263, 5965.0767, 77.23391, 0.04252738, 0.028726239260970492, 0.16948818, 209.30631160736084, -168.55824, 322.0195, 263316.12, 513.1434, 0.040522102, 0.004169625723129048, 0.06457264, 4.04609814286232],
        [-2.624227, 17.21473, 3585.543, 59.879402, 0.037777375, 0.017267047456248998, 0.13140413, 331.00085258483887, -26.609068, 222.3025, 120319.54, 346.87106, 0.027973974, 0.001905266702029852, 0.04364936, 3.09450663626194],
        [-7.626922, 25.303167, 6272.219, 79.19734, 0.05552724, 0.030205332806813555, 0.17379682, 3048.2200622558594, -147.8363, 318.08496, 327443.6, 572.22687, 0.040026993, 0.005185087815544217, 0.07200755, 4.281115159392357],
        [3.5159266, 23.52587, 5087.252, 71.324974, 0.051626995, 0.024498843534813984, 0.15652107, 829.193115234375, 11.684146, 253.77457, 128860.09, 358.9709, 0.031934336, 0.0020405068682551085, 0.045171972, 3.6780644208192825],
    ])
    result_baseline = [-14.172823, 26.664255, 6433.2803, 80.20773, 0.05851412, 0.030980959739528607, 0.1760141, 1427.5680541992188, -210.35655, 326.40854, 208159.39, 456.24487, 0.040336978, 0.003178920740372834, 0.05638192, 4.810509085655212]

    if transform == 'exp':
        results = results_exponentiation
        denominators = np.arange(4.00, 1.25-0.01, -0.25)
        labels = [f"1/{denominator}" for denominator in denominators]

    elif transform == 'log':
        results = results_logarithm
        x1 = [1e-10, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0]
        labels = [str(_) for _ in x1]

    else:
        raise ValueError(f"Invalid input argument: {transform}.")

    me = results[:, 0]
    mae = results[:, 1]
    mse = results[:, 2]
    rmse = results[:, 3]
    nmae = results[:, 4]
    nmse = results[:, 5]
    nrmse = results[:, 6]
    mre = results[:, 7]
    maxima_me = results[:, 0+8]
    maxima_mae = results[:, 1+8]
    maxima_mse = results[:, 2+8]
    maxima_rmse = results[:, 3+8]
    maxima_nmae = results[:, 4+8]
    maxima_nmse = results[:, 5+8]
    maxima_nrmse = results[:, 6+8]
    maxima_mre = results[:, 7+8]

    plt.figure(1, figsize=(6, 6))

    plt.subplot(2, 2, 1)
    plt.plot(mae, '.-')
    plt.axhline(result_baseline[1], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(mae)), labels=labels, rotation=90)
    plt.legend()
    plt.title('MAE')

    plt.subplot(2, 2, 2)
    plt.semilogy(mre, '.-')
    plt.axhline(result_baseline[7], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(mre)), labels=labels, rotation=90)
    plt.legend()
    plt.title('MRE')

    plt.subplot(2, 2, 3)
    plt.semilogy(mse, '.-')
    plt.axhline(result_baseline[2], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(mse)), labels=labels, rotation=90)
    plt.legend()
    plt.title('MSE')

    plt.subplot(2, 2, 4)
    plt.semilogy(rmse, '.-')
    plt.axhline(result_baseline[3], linestyle='--', color=[0.0]*3, label='Baseline')
    plt.xticks(ticks=range(len(rmse)), labels=labels, rotation=90)
    plt.legend()
    plt.title('RMSE')

    plt.tight_layout()

    # plt.figure(2, figsize=(6, 6))

    # plt.subplot(2, 2, 1)
    # plt.plot(maxima_mae, '.-')
    # plt.axhline(result_baseline[1+8], linestyle='--', color=[0.0]*3, label='Baseline')
    # plt.xticks(ticks=range(len(maxima_mae)), labels=labels, rotation=90)
    # plt.legend()
    # plt.title('MAE (Maxima)')

    # plt.subplot(2, 2, 2)
    # plt.semilogy(maxima_mre, '.-')
    # plt.axhline(result_baseline[7+8], linestyle='--', color=[0.0]*3, label='Baseline')
    # plt.xticks(ticks=range(len(maxima_mre)), labels=labels, rotation=90)
    # plt.legend()
    # plt.title('MRE (Maxima)')

    # plt.subplot(2, 2, 3)
    # plt.plot(maxima_mse, '.-')
    # plt.axhline(result_baseline[2+8], linestyle='--', color=[0.0]*3, label='Baseline')
    # plt.xticks(ticks=range(len(maxima_mse)), labels=labels, rotation=90)
    # plt.legend()
    # plt.title('MSE (Maxima)')

    # plt.subplot(2, 2, 4)
    # plt.plot(maxima_rmse, '.-')
    # plt.axhline(result_baseline[3+8], linestyle='--', color=[0.0]*3, label='Baseline')
    # plt.xticks(ticks=range(len(maxima_rmse)), labels=labels, rotation=90)
    # plt.legend()
    # plt.title('RMSE (Maxima)')

    # plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    plot_page_progress(current=122, previous=113, goal_1=120, goal_2=150)
    plot_page_history()
    plot_page_distribution()