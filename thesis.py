"""
Functions for generating figures.
"""


import matplotlib.pyplot as plt

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

def plot_label_transformation_histograms():
    """Comparison of raw histogram and average sample histogram (both normalized)."""

    data = read_pickle('Labels 2D/labels.pickle')
    data /= data.max()

    transformed = data.copy()
    transformed /= transformed.max(axis=tuple(range(1, data.ndim)), keepdims=True)

    plt.figure()
    plt.hist(data.flatten(), bins=100, alpha=0.5, label='Raw')
    plt.hist(transformed.flatten(), bins=100, alpha=0.5, label='Average Sample')
    plt.legend()
    plt.show()

def plot_label_transformation_metrics():
    """Evaluation metrics for models trained with different transformation exponents."""
    denominators = np.arange(1.0, 6.0_1, 0.2).round(2)

    def _results():
        return np.array([
            [32.906967, 46.269577, 17178.643, 131.06732, 0.10153756, 0.08272775496663787, 0.28762433, 13.928377628326416, 383.117, 525.7178, 540846.4, 735.4226, 0.06496725, 0.008259573366739449, 0.0908822, 6.58290833234787],
            [1.1472228, 24.602371, 8779.089, 93.69679, 0.053989362, 0.04227775312336138, 0.20561555, 11.056730151176453, 112.57243, 373.93912, 395543.16, 628.92224, 0.04621072, 0.006040565065742671, 0.077721074, 4.612907767295837],
            [12.754568, 20.858007, 8414.035, 91.72805, 0.045772437, 0.040519751706441336, 0.20129518, 8.25856328010559, 268.88104, 354.90518, 310343.2, 557.08453, 0.043858543, 0.004739427764536699, 0.0688435, 3.892902657389641],
            [-15.984247, 21.795622, 5653.059, 75.18683, 0.04783001, 0.02722362649217145, 0.16499583, 6.195186823606491, -291.54562, 347.88626, 214644.53, 463.29745, 0.042991158, 0.003277959020486683, 0.05725346, 5.264812335371971],
            [4.4796734, 15.306235, 2784.804, 52.77124, 0.0335892, 0.01341087573669464, 0.115805335, 5.324644967913628, -63.761665, 244.36684, 118153.15, 343.73413, 0.030198412, 0.0018043840971122117, 0.042478044, 3.4500878304243088],
            [-7.3383665, 18.388672, 9561.428, 97.782555, 0.040353533, 0.04604528251538518, 0.21458165, 4.459375515580177, -259.31165, 397.67572, 515434.88, 717.9379, 0.04914405, 0.00787149986137686, 0.08872148, 4.850877076387405],
            [-2.0301495, 13.915469, 4092.506, 63.9727, 0.030537192, 0.0197084217385682, 0.14038669, 3.7334833294153214, 71.79058, 315.59906, 188579.94, 434.2579, 0.039001163, 0.0028799117480936983, 0.053664807, 4.340257868170738],
            [-12.966896, 21.52523, 8803.932, 93.82927, 0.047236644, 0.04239738930180533, 0.20590627, 3.809420019388199, -195.98631, 435.42715, 444664.06, 666.83136, 0.0538093, 0.006790718432329634, 0.08240581, 5.0878554582595825],
            [-10.441313, 22.769466, 7944.3667, 89.13118, 0.049967088, 0.03825794841875219, 0.1955964, 3.9312515407800674, -490.11383, 535.839, 509614.78, 713.8731, 0.06621802, 0.007782617891280588, 0.08821915, 7.466008514165878],
            [-0.92448443, 20.544, 7664.713, 87.54835, 0.045083355, 0.0369112153041802, 0.19212292, 4.302410408854485, -198.52142, 387.49405, 299627.56, 547.38245, 0.047885817, 0.004575783345439006, 0.06764454, 5.274036526679993],
            [-38.271317, 87.45562, 134371.03, 366.56656, 0.19191945, 0.6470951750191363, 0.8044223, 12.617471814155579, -1085.2491, 1463.9175, 4288582.0, 2070.8892, 0.18090828, 0.06549338094071204, 0.25591674, 21.39517068862915],
            [-14.198751, 27.911934, 19085.752, 138.1512, 0.061252125, 0.09191189671900805, 0.30316976, 3.6336392164230347, 80.18582, 553.8853, 644688.9, 802.9252, 0.06844815, 0.009845411392067143, 0.099224046, 7.08104744553566],
            [3.1707206, 24.491652, 11025.677, 105.00322, 0.053746406, 0.05309675033217913, 0.23042731, 3.6325544118881226, -150.68959, 462.36536, 382660.25, 618.5954, 0.057138283, 0.00584382426944527, 0.07644491, 6.796461343765259],
            [-11.847596, 21.086416, 13190.383, 114.84939, 0.04627368, 0.06352137804720535, 0.25203446, 2.9047315940260887, -377.66367, 499.16302, 546128.5, 739.00507, 0.061685666, 0.008340239709321089, 0.09132491, 6.543885171413422],
            [-14.23098, 24.224485, 8013.3623, 89.51739, 0.053160105, 0.038590223166449854, 0.19644395, 4.106902703642845, -254.22583, 463.25406, 370261.34, 608.491, 0.0572481, 0.005654472092136441, 0.07519622, 7.0898473262786865],
            [2.2343209, 27.009151, 14692.812, 121.213905, 0.059270985, 0.0707566564677079, 0.26600122, 3.71566079556942, -85.97129, 571.7556, 851675.94, 922.8629, 0.07065652, 0.013006428841217344, 0.11404573, 7.271084189414978],
            [-34.663036, 41.556408, 52567.773, 229.27663, 0.091194615, 0.25315235824377313, 0.5031425, 3.8928214460611343, -591.30865, 824.5082, 2181079.5, 1476.8478, 0.101891235, 0.033308508629537166, 0.18250619, 8.94339382648468],
            [0.47563648, 19.378725, 6056.4854, 77.823425, 0.042526197, 0.029166435554584112, 0.17078184, 3.465825691819191, -27.927855, 501.41547, 587263.7, 766.3313, 0.061964016, 0.008968438611663429, 0.09470184, 6.790686398744583],
            [1.7484161, 22.299286, 8829.334, 93.96454, 0.048935287, 0.04251971452669387, 0.2062031, 2.9925793409347534, 15.258085, 458.94693, 379378.56, 615.93713, 0.05671583, 0.005793706338027869, 0.0761164, 6.69747069478035],
            [-19.870981, 28.364588, 14769.587, 121.53019, 0.062245466, 0.07112640529887738, 0.26669535, 3.229963406920433, -534.80927, 678.6859, 913591.3, 955.8197, 0.08387078, 0.01395197383510158, 0.11811847, 9.002934396266937],
            [-34.825436, 41.354855, 41735.62, 204.29298, 0.090752326, 0.2009876592071813, 0.44831648, 3.9566904306411743, -740.72943, 875.04596, 1631514.0, 1277.3074, 0.1081366, 0.024915780533543458, 0.15784734, 10.771247744560242],
            [4.593722, 26.580206, 13437.372, 115.91968, 0.05832967, 0.06471079533629184, 0.25438318, 3.8233477622270584, -249.58394, 570.6688, 592859.94, 769.974, 0.07052223, 0.009053902136224569, 0.095152, 8.315567672252655],
            [1.1025319, 20.57938, 6333.8584, 79.58554, 0.045160994, 0.030502170540910714, 0.1746487, 3.160422667860985, -175.96272, 538.56384, 494732.72, 703.3724, 0.06655475, 0.007555345429544755, 0.08692149, 7.543735206127167],
            [-0.240402, 22.705067, 9977.296, 99.88641, 0.04982577, 0.04804799896825759, 0.21919854, 3.0967099592089653, -160.44588, 840.6402, 1313308.8, 1145.9968, 0.1038848, 0.020056286729860907, 0.14162022, 10.351143777370453],
            [-4.3387294, 31.962133, 22327.518, 149.42395, 0.07014021, 0.10752341952653952, 0.32790762, 3.686082363128662, -136.5794, 693.536, 834037.75, 913.25665, 0.08570593, 0.01273706602314806, 0.11285861, 9.524773061275482],
            [44.376015, 53.045895, 79025.68, 281.11505, 0.11640806, 0.38056667090131463, 0.6169008, 4.012053832411766, 644.3008, 897.736, 1939977.6, 1392.8308, 0.1109406, 0.02962650442747342, 0.17212352, 9.751355648040771],
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
    plt.plot(denominators, mre, '.-')
    plt.xticks(ticks=denominators, labels=labels, rotation=90)
    plt.title('MRE')
    plt.show()


if __name__ == '__main__':
    plot_page_progress(current=21, previous=2, goal_1=120, goal_2=150)

    # plot_label_transformation_metrics()