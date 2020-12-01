# K-means implementation in Python
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from copy import deepcopy


class K_Means:
    def __init__(self, data, core_number):
        self.data = data
        self.core_number = core_number
        self.distance = np.zeros((len(self.data), self.core_number))

    def Euclidean_operator(self, x, y):
        # input format: tuple
        if len(x) != len(y): raise Exception('Dimension error!')
        distance = 0
        for dim in range(len(x)):
            distance += pow((x[dim] - y[dim]), 2)
        distance = math.sqrt(distance)
        return distance

    def Manhattan_operator(self, x, y):
        if len(x) != len(y): raise Exception('Dimension error!')
        distance = 0
        for dim in range(len(x)):
            distance += abs(x[dim] - y[dim])
        return distance

    def square_error(self):
        # calculate cluster square error using distance data
        s_error = np.zeros(len(self.clusters))
        for index in range(len(self.clusters)):
            s_error[index] = np.sum(np.square(self.distance[:, index]))
        return np.sum(s_error)

    def k_means_operation(self, operator):
        # initial
        self.working_operator = operator
        self.clusters = {}
        self.core_history = {}
        self.core = [self.data[random.randrange(0, len(self.data), 1)]
                         for i in range(self.core_number)]
        for core in self.core:
            self.clusters[core] = []
        exit_condition = False
        runtime = 0
        while not exit_condition:
            # iteration and objects reassign
            runtime += 1
            for obj_ind, obj in enumerate(self.data):
                for core_ind, core in enumerate(self.core):
                    self.distance[obj_ind, core_ind] = operator(obj, core)
                cluster_index = min([core_ind for core_ind in range(len(self.core))],
                                    key=lambda ind:self.distance[obj_ind, ind])
                self.clusters[self.core[cluster_index]].append(self.data[obj_ind])

            # check exit condition
            exit_condition = True
            # check objects assignment changes
            if runtime == 1:
                exit_condition = False
            else:
                for core, obj_list in self.clusters.items():
                    if set(obj_list) != set(cluster_history[self.core_history[core]]):
                        exit_condition = False
                        break
            # update cores
            if exit_condition is False:
                cluster_history = deepcopy(self.clusters)
                update_clusters = {}
                update_core = []
                update_history = {}
                delete_index = 0        # mark redundant cluster index for delete
                for core, obj_list in self.clusters.items():
                    # k is larger than need, delete redundant clusters
                    if len(obj_list) == 0:
                        self.distance[:, delete_index] = 0      # delete distance data
                        break
                    delete_index += 1
                    # calculate new cores
                    new_core = tuple(np.sum([np.array(obj) for obj in obj_list], axis=0)/len(obj_list))
                    update_core.append(new_core)
                    update_clusters[new_core] = []
                    update_history[new_core] = core
                self.clusters = update_clusters
                self.core = update_core
                self.core_history = update_history

            # calculate square error of each cluster
            s_errors = self.square_error()
            print(s_errors)

        print('=============== final clusters ==============')
        for core, obj_list in self.clusters.items():
            print('core: {}    nodes({}): {}'.format(core, len(obj_list), obj_list))
            x_coord = []
            y_coord = []
            for obj in obj_list:
                x_coord.append(obj[0])
                y_coord.append(obj[1])
            plt.plot(x_coord, y_coord, '.')
        plt.show()
        s_errors = self.square_error()
        print('=============== final error ================')
        print(s_errors)

    def mean_cluster_dis(self, node, cluster, distance_matrix):
        '''
            mean(distance between node and others in cluster)
        :param node: tuple
        :param cluster: list of nodes
        :param distance_matrix: dict[node_x][node_y] = distance between x and y
        :return:
        '''
        N = len(cluster)
        param_node = np.zeros(N)
        for y_node_idx, y_node in enumerate(cluster):
            param_node[y_node_idx] = distance_matrix[node][y_node]
        param_node = np.mean(param_node)
        return param_node

    def Silhouette_Coefficient(self):
        '''
            Calculate Silhouette_Coefficient on dataset with k
        '''
        # using same distance operator with k-means operation
        try:
            operator = self.working_operator
        except:
            raise Exception('Run k-means operation first.')
        # calculate all distance between nodes
        all_conp_distance = {}
        for node in self.data:
            all_conp_distance[node] = {}
        for node_x in self.data:
            for node_y in self.data:
                if all_conp_distance.get(node_x).get(node_y) is not None:
                    continue
                else:
                    node_distance = operator(node_x, node_y)
                    all_conp_distance[node_x][node_y] = node_distance
                    all_conp_distance[node_y][node_x] = node_distance
        # calculate parameter a and b for each node
        a_dataset = []
        b_dataset = []
        for cls_core, cluster in self.clusters.items():
            for node in cluster:
                # a = mean(sum of distance between node and others in the same cluster)
                a_dataset.append(self.mean_cluster_dis(node, cluster, all_conp_distance))
                # b - mean(sum of distance between node and others in the closest cluster)
                closest_cluster = min([i for i in range(self.core_number) if self.core[i]!=cls_core],
                                      key=lambda i:operator(self.core[i], node))
                b_dataset.append(self.mean_cluster_dis(node, self.clusters[self.core[closest_cluster]], all_conp_distance))
        a_dataset = np.array(a_dataset)
        b_dataset = np.array(b_dataset)
        # s = (b-a)/max(a, b)
        s_dataset = (b_dataset-a_dataset)/np.maximum(a_dataset, b_dataset)
        s_dataset = np.mean(s_dataset)

        print('Silhouette_Coefficient on dataset with k as {}: {}'.format(self.core_number, s_dataset))

        return s_dataset





