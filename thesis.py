"""
Functions for generating figures.
"""


import matplotlib.pyplot as plt
from PIL import Image

from helpers import *
from datasets import CantileverDataset


def plot_page_progress(current: int, previous: int, goal_1: int=120, goal_2: int=150):
    """Figure showing a bar graph of the current number of pages compared to the desired number of pages."""

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

def plot_label_histogram():
    """Figure of histogram of all labels in dataset."""

    labels = read_pickle('Labels 2D/labels.pickle')
    print(f"Mean {labels.mean()}, median {np.median(labels)}")

    plt.figure()
    plt.hist(labels[labels != -1], bins=100)
    plt.xlabel('Stress [Pa]')
    plt.xticks([0, 20_000, 40_000, labels.max()])
    plt.show()

def plot_input():
    """Figure of both channels of an input."""

    samples = read_samples('samples.csv')
    inputs = generate_input_images(samples, is_3d=False)
    i = 5200  #829

    plt.figure(figsize=(4, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(inputs[i, 0, ...], cmap='gray', vmin=0, vmax=2)
    plt.title(f"Length = {samples['Length'][i]} m, Height = {samples['Height'][i]} m")
    plt.colorbar(fraction=0.023, ticks=[0, 1, 2])
    plt.subplot(2, 1, 2)
    plt.imshow(inputs[i, 1, ...], cmap='gray', vmin=0, vmax=2)
    plt.title(f"Angle = {int(samples['Angle XY'][i])}°")
    plt.colorbar(fraction=0.023, ticks=[0, 1, 2])
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

def plot_inputs_labels():
    """Many input images with corresponding labels."""

    samples = read_samples('samples.csv')
    dataset = CantileverDataset(samples, is_3d=False)

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

def plot_stratified_sampling_predictions():
    import glob

    folders = glob.glob('Stratified Sampling Predictions/*')

    for folder in folders:
        plt.figure()

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
                plt.ylabel("Normal")
            elif i == 4:
                plt.ylabel("Sampling")

            # Show label.
            if 's' in file:
                plt.subplot(3, 4, 1 + i%4)
                plt.imshow(true)
                plt.xticks([])
                plt.yticks([])
                if i == 0:
                    plt.ylabel('True')
    
            plt.suptitle(f"{os.path.basename(folder)} Data", fontweight='bold')
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

def plot_lt_metrics_exponentiation():
    """Evaluation metrics for models trained with different transformation exponents."""
    denominators = np.arange(1.00, 4.01, 0.25)

    def _results():
        return np.array([
            [-14.172823, 26.664255, 6433.2803, 80.20773, 0.05851412, 0.030980959739528607, 0.1760141, 1427.5680541992188, -210.35655, 326.40854, 208159.39, 456.24487, 0.040336978, 0.003178920740372834, 0.05638192, 4.810509085655212],
            [4.60984, 20.740469, 5251.4233, 72.466705, 0.0455145, 0.02528945575867015, 0.1590266, 397.6400136947632, 7.3831654, 272.31485, 138784.64, 372.53812, 0.03365218, 0.0021194593777553853, 0.04603759, 3.94560769200325],
            [0.032291014, 18.51287, 5554.6357, 74.52943, 0.04062609, 0.026749649491570528, 0.1635532, 98.50154519081116, -95.56946, 367.6348, 281610.88, 530.6702, 0.045431644, 0.004300640238060562, 0.065579265, 5.399021133780479],
            [-1.7691576, 20.721878, 5505.4775, 74.198906, 0.045473717, 0.026512927303431896, 0.16282791, 61.136698722839355, -56.847607, 331.50977, 205032.03, 452.80463, 0.04096738, 0.003131161052232236, 0.055956777, 4.9522023648023605],
            [-18.129675, 23.837383, 7206.758, 84.892624, 0.05231061, 0.03470582100448338, 0.18629499, 11.843579262495041, -255.76085, 401.44812, 277615.3, 526.89215, 0.049610235, 0.004239621724975135, 0.06511238, 5.811024084687233],
            [-0.5516696, 17.674067, 4542.057, 67.39478, 0.03878535, 0.02187333707098249, 0.14789636, 11.698953807353973, -43.0056, 379.8246, 254243.0, 504.22516, 0.046938036, 0.003882689814607591, 0.062311236, 5.8918967843055725],
            [-24.597153, 30.456278, 18258.375, 135.12355, 0.066835634, 0.08792746973650914, 0.29652566, 5.7304747402668, -611.0491, 660.9262, 779511.06, 882.89923, 0.081676066, 0.011904357888880682, 0.10910709, 8.970752358436584],
            [20.782646, 30.892614, 13221.378, 114.98425, 0.06779317, 0.06367064229407914, 0.25233042, 6.047312170267105, 110.81973, 467.2452, 342068.3, 584.8661, 0.057741318, 0.0052239202371107815, 0.0722767, 6.967458873987198],
            [-16.685993, 24.093767, 8496.485, 92.176384, 0.052873246, 0.04091681536169548, 0.20227906, 4.978133365511894, -236.27664, 443.35367, 346819.16, 588.9135, 0.054788847, 0.005296473080803301, 0.07277687, 6.800913065671921],
            [0.63365155, 19.331297, 5920.115, 76.94228, 0.04242211, 0.028509701593298933, 0.16884816, 4.157403111457825, -143.24911, 433.18835, 309909.2, 556.6949, 0.053532634, 0.004732799903083129, 0.068795346, 7.103805243968964],
            [3.702548, 37.078655, 36211.992, 190.2945, 0.08136831, 0.17438737711186747, 0.41759717, 4.9101874232292175, -293.04633, 682.8758, 835414.75, 914.01025, 0.084388554, 0.012758094975271481, 0.11295174, 9.729297459125519],
            [-11.426707, 28.28825, 19424.354, 139.37128, 0.062077958, 0.09354256175271285, 0.3058473, 4.718296602368355, -886.58777, 930.30676, 1676946.4, 1294.9696, 0.11496563, 0.025609604236323604, 0.16003, 11.780910193920135],
            [7.9987, 24.679495, 10782.223, 103.83748, 0.054158606, 0.05192430483639175, 0.22786905, 4.296436160802841, -136.7292, 555.6768, 519277.44, 720.6091, 0.068669535, 0.00793018182422635, 0.08905157, 8.499367535114288],
        ])

    results = _results()

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

    labels = [f"1/{denominator}" for denominator in denominators]
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(denominators, mae, '.-')
    plt.xticks(ticks=denominators, labels=labels, rotation=90)
    plt.title('MAE')
    plt.subplot(2, 2, 2)
    plt.plot(denominators, mse, '.-')
    plt.xticks(ticks=denominators, labels=labels, rotation=90)
    plt.title('MSE')
    plt.subplot(2, 2, 3)
    plt.plot(denominators, rmse, '.-')
    plt.xticks(ticks=denominators, labels=labels, rotation=90)
    plt.title('RMSE')
    plt.subplot(2, 2, 4)
    plt.semilogy(denominators, mre, '.-')
    plt.xticks(ticks=denominators, labels=labels, rotation=90)
    plt.title('MRE')
    plt.show()

def plot_lt_metrics_logarithm():
    """Evaluation metrics for models trained with different transformation logarithms."""
    size = [0.5, 1]
    x_1 = [1e-2, 0.1, 0.5, 1.0]

    def _results():
        return np.array([
            [-25.909452, 31.70677, 24274.783, 155.80367, 0.06957981, 0.11690090445277997, 0.34190774, 286.30990982055664, -446.5486, 611.2797, 895812.4, 946.47363, 0.076921865, 0.01418523959327339, 0.1191018, 7.417932152748108],
            [-9.971659, 25.07012, 9697.933, 98.47808, 0.055015903, 0.046702791194708215, 0.21610828, 962.7808570861816, -264.52115, 388.84775, 376059.62, 613.237, 0.0489316, 0.005954925418374069, 0.07716816, 5.185775458812714],
            [6.571792, 30.303213, 9377.879, 96.83945, 0.06649973, 0.04516136037727199, 0.21251202, 1835.4372024536133, -169.67236, 360.88824, 277625.34, 526.9017, 0.045413245, 0.0043962130108535676, 0.066303946, 5.172388255596161],
            [-10.520536, 28.46741, 6201.72, 78.751, 0.062471077, 0.029865808336069468, 0.17281726, 2976.873016357422, -124.54792, 288.80194, 166407.17, 407.93036, 0.03634209, 0.0026350669726859935, 0.051332906, 4.436640068888664],
            [-2.1913984, 31.808561, 27684.307, 166.38602, 0.06980319, 0.133320263186554, 0.36513048, 301.25558376312256, -441.36874, 750.15967, 1359901.5, 1166.1482, 0.094398156, 0.021534117119951453, 0.14674509, 9.652598202228546],
            [1.0918914, 27.39456, 8335.709, 91.3001, 0.060116805, 0.04014261825032946, 0.20035623, 1426.124095916748, -460.6867, 516.3282, 540647.75, 735.28754, 0.06497341, 0.008561187680974124, 0.09252669, 6.971164047718048],
            [-16.57666, 30.9987, 18189.473, 134.86835, 0.06802596, 0.0875956426728829, 0.2959656, 1016.8990135192871, -230.46414, 403.5637, 514281.72, 717.1344, 0.050783414, 0.008143680085809474, 0.09024234, 4.912705719470978],
            [-16.004984, 30.58923, 7240.871, 85.09331, 0.067127384, 0.034870096786256, 0.18673536, 1608.900260925293, -228.02551, 329.22263, 193880.89, 440.3191, 0.041428525, 0.0030701148620243804, 0.05540862, 5.054793134331703],
        ])

    results = _results()

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
    print(mse)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.imshow(mae.reshape((len(size), len(x_1))), cmap='Reds')
    plt.colorbar()
    plt.xticks(ticks=range(len(x_1)), labels=x_1)
    plt.yticks(ticks=range(len(size)), labels=size)
    plt.xlabel('Size')
    plt.ylabel('x_1')
    plt.title('MAE')

    plt.subplot(2, 2, 2)
    plt.imshow(mse.reshape((len(size), len(x_1))), cmap='Reds')
    plt.colorbar()
    plt.xticks(ticks=range(len(x_1)), labels=x_1)
    plt.yticks(ticks=range(len(size)), labels=size)
    plt.xlabel('Size')
    plt.ylabel('x_1')
    plt.title('MSE')

    plt.subplot(2, 2, 3)
    plt.imshow(rmse.reshape((len(size), len(x_1))), cmap='Reds')
    plt.colorbar()
    plt.xticks(ticks=range(len(x_1)), labels=x_1)
    plt.yticks(ticks=range(len(size)), labels=size)
    plt.xlabel('Size')
    plt.ylabel('x_1')
    plt.title('RMSE')

    plt.subplot(2, 2, 4)
    plt.imshow(mre.reshape((len(size), len(x_1))), cmap='Reds')
    plt.colorbar()
    plt.xticks(ticks=range(len(x_1)), labels=x_1)
    plt.yticks(ticks=range(len(size)), labels=size)
    plt.xlabel('Size')
    plt.ylabel('x_1')
    plt.title('MRE')

    plt.show()


if __name__ == '__main__':
    # plot_page_progress(current=39, previous=33, goal_1=120, goal_2=150)

    plot_lt_logarithms_histogram()