from kMeans_Module import K_Means
from matplotlib import pyplot as plt
import pickle
import numpy as np

test_data = [
     (23, 15), (29, 25), (55, 59), (53, 61), (59, 68),
    (55, 59), (21, 26), (27, 24), (17, 15),  (17, 21),
    (15, 25), (17, 5), (13, 16), (13, 9), (13, 18),
    (17, 17), (19, 6),  (13, 16), (56, 99), (62, 104),
    (54, 101), (45, 59), (57, 64), (19, 14), (11, 7),
    (15, 13), (45, 66), (45, 69), (57, 68), (23, 20),
    (19, 21), (17, 20), (47, 66), (59, 99), (62, 93)
]

def data_load():
    dataSet, dataLabel = pickle.load(open('mnist.pkl', 'rb'), encoding='latin1')
    input_data = []
    for row in dataSet:
        input_data.append(tuple(row))
    return input_data, dataSet, dataLabel

def multi_k_run(data, k_range):
    '''
        run k-means with multi k value
    :param k_range: include min and max
    '''
    Silhouette_Coefficients = []
    for k in range(k_range[0], k_range[1]+1):
        k_means = K_Means(data, k)
        try:
            # k_means.k_means_operation(k_means.Euclidean_operator)
            k_means.k_means_operation(k_means.Manhattan_operator)
            Silhouette_Coefficients.append(k_means.Silhouette_Coefficient())
        except:
            Silhouette_Coefficients.append(0)
    ks = range(k_range[0], k_range[1]+1)
    plt.plot(ks, Silhouette_Coefficients)
    plt.show()

def label_plot(dataset, data_label):
    '''
        draw mnist dataset label picture.
    '''
    for number in range(0, 10):
        number_data_index = np.where(data_label==number)
        number_data = dataset[number_data_index, :]
        x_axis = number_data[0, :, 0]
        y_axis = number_data[0, :, 1]
        plt.plot(x_axis, y_axis, '.')
        plt.text(x_axis[0]+0.5, y_axis[0]+0.5, str(number))
    plt.show()


if __name__ == '__main__':
    input_data, dataset, data_label = data_load()
    label_plot(dataset, data_label)
    multi_k_run(input_data, [9, 11])

    pass
