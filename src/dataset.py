import glob
import random
import pickle
import torch
import numpy as np
from utils import get_data_from_path


class Dataset:

    def __init__(self, args):
        self.args = args

        '''
        init dataset
        '''
        training_paths = glob.glob(args.training_root_path + "*.gexf")
        test_paths = glob.glob(args.test_root_path + "*.gexf")
        random.shuffle(training_paths)
        random.shuffle(test_paths)
        if self.args.small_dataset:
            training_paths = training_paths[:16]
            test_paths = test_paths[:8]

        self.training_graphs = []
        for path in training_paths:
            self.training_graphs.append(get_data_from_path(path, self.args.current_dataset_name))

        self.test_graphs = []
        for path in test_paths:
            self.test_graphs.append(get_data_from_path(path, self.args.current_dataset_name))

        self.ged_dict = pickle.load(open(args.ged_path, "rb"))

        self.training_graph_index_pairs = []
        for i in range(len(self.training_graphs)):
            for j in range(len(self.training_graphs)):
                self.training_graph_index_pairs.append((i, j))

        self.test_graph_index_pairs = []
        for i in range(len(self.test_graphs)):
            for j in range(len(self.training_graphs)):
                self.test_graph_index_pairs.append((i, j))

        if self.args.validate:
            cut_num = len(self.test_graphs) * len(self.training_graphs)
            self.validation_graph_index_pairs = self.training_graph_index_pairs[:cut_num]
            self.training_graph_index_pairs = self.training_graph_index_pairs[cut_num:]

        """
        init labels
        """
        tar_node_set = set()
        tar_edge_set = set()

        edge_count_dict = dict()
        total_edge_num = 0

        if self.args.cut_edge_ratio == 1:
            for graph in self.training_graphs + self.test_graphs:
                tar_node_set |= set(graph["labels"])
                total_edge_num += len(graph["graph"])
                for edge in graph["graph"]:
                    cur_edge_label = tuple(sorted([graph["labels"][edge[0]], graph["labels"][edge[1]]]))
                    tar_edge_set.add(cur_edge_label)
        else:
            for graph in self.training_graphs + self.test_graphs:
                tar_node_set |= set(graph["labels"])
                total_edge_num += len(graph["graph"])
                for edge in graph["graph"]:
                    cur_edge_label = tuple(sorted([graph["labels"][edge[0]], graph["labels"][edge[1]]]))
                    edge_count_dict[cur_edge_label] = 1 if not edge_count_dict.get(cur_edge_label) else edge_count_dict[cur_edge_label] + 1

        if self.args.cut_edge_ratio != 1:
            edge_rank = sorted(edge_count_dict.items(), key=lambda k_v: (k_v[1], k_v[0]), reverse=True)
            temp_edge_count = 0
            cut_edge_count = total_edge_num * self.args.label_cut_ratio
            for e in edge_rank:
                tar_edge_set.add(e[0])
                temp_edge_count += e[1]
                if temp_edge_count >= cut_edge_count:
                    break
            tar_edge_set.add(("Z-Others", "Z-Others"))

        self.global_node_labels = sorted(tar_node_set)
        self.global_edge_labels = sorted(tar_edge_set)
        self.global_node_labels = {val: index for index, val in enumerate(self.global_node_labels)}
        self.global_edge_labels = {val: index for index, val in enumerate(self.global_edge_labels)}

        self.number_of_node_labels = len(self.global_node_labels)
        self.number_of_edge_labels = len(self.global_edge_labels)

    def get_training_graphs(self):
        return self.training_graphs

    def get_test_graphs(self):
        return self.test_graphs

    def get_data(self, graph_index_pair, mode="training"):
        data = dict()
        if mode == "training" or mode == "validation":
            g_1 = self.training_graphs[graph_index_pair[0]]
            g_2 = self.training_graphs[graph_index_pair[1]]
        else:
            g_1 = self.test_graphs[graph_index_pair[0]]
            g_2 = self.training_graphs[graph_index_pair[1]]

        graph_pair = dict()
        graph_pair["graph_1"] = g_1.get("graph")
        graph_pair["graph_2"] = g_2.get("graph")
        graph_pair["labels_1"] = g_1.get("labels")
        graph_pair["labels_2"] = g_2.get("labels")
        graph_pair["ged"] = self.ged_dict.get((g_1.get("id"), g_2.get("id")))

        data["graph_1"] = graph_pair["graph_1"]
        data["graph_2"] = graph_pair["graph_2"]

        trans_edge_index_1 = []
        for edge_index_1, edge_1 in enumerate(graph_pair["graph_1"]):
            for edge_index_2, edge_2 in enumerate(graph_pair["graph_1"]):
                if edge_1[0] == edge_2[0] or edge_1[0] == edge_2[1] or edge_1[1] == edge_2[0] or edge_1[1] == edge_2[1]:
                    trans_edge_index_1.append([edge_index_1, edge_index_2])
        data["trans_edge_index_1"] = trans_edge_index_1

        trans_edge_index_2 = []
        for edge_index_1, edge_1 in enumerate(graph_pair["graph_2"]):
            for edge_index_2, edge_2 in enumerate(graph_pair["graph_2"]):
                if edge_1[0] == edge_2[0] or edge_1[0] == edge_2[1] or edge_1[1] == edge_2[0] or edge_1[1] == edge_2[1]:
                    trans_edge_index_2.append([edge_index_1, edge_index_2])
        data["trans_edge_index_2"] = trans_edge_index_2

        # process node labels
        data["node_labels_1"] = graph_pair["labels_1"]
        data["node_labels_2"] = graph_pair["labels_2"]

        # process edge labels
        edge_labels_1 = []
        edge_labels_2 = []
        for edge in graph_pair["graph_1"]:
            edge_labels_1.append(tuple(sorted([graph_pair["labels_1"][edge[0]], graph_pair["labels_1"][edge[1]]])))
        for edge in graph_pair["graph_2"]:
            edge_labels_2.append(tuple(sorted([graph_pair["labels_2"][edge[0]], graph_pair["labels_2"][edge[1]]])))
        data["edge_labels_1"] = edge_labels_1
        data["edge_labels_2"] = edge_labels_2

        data["ged"] = graph_pair["ged"]
        return data

    def transfer_to_torch(self, data):
        new_data = dict()

        edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
        edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

        new_data["edge_index_1"] = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long) \
            .to(self.args.device)
        new_data["edge_index_2"] = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long) \
            .to(self.args.device)

        trans_edges_1 = data["trans_edge_index_1"] + [[y, x] for x, y in data["trans_edge_index_1"]]
        trans_edges_2 = data["trans_edge_index_2"] + [[y, x] for x, y in data["trans_edge_index_2"]]
        new_data["trans_edge_index_1"] = torch.from_numpy(np.array(trans_edges_1, dtype=np.int64).T).type(torch.long) \
            .to(self.args.device)
        new_data["trans_edge_index_2"] = torch.from_numpy(np.array(trans_edges_2, dtype=np.int64).T).type(torch.long) \
            .to(self.args.device)

        node_features_1, node_features_2 = [], []
        for n in data["node_labels_1"]:
            node_features_1.append(
                [1.0 if self.global_node_labels[n] == i else 0.0 for i in self.global_node_labels.values()])

        for n in data["node_labels_2"]:
            node_features_2.append(
                [1.0 if self.global_node_labels[n] == i else 0.0 for i in self.global_node_labels.values()])

        new_data["node_features_1"] = torch.FloatTensor(np.array(node_features_1)).to(self.args.device)
        new_data["node_features_2"] = torch.FloatTensor(np.array(node_features_2)).to(self.args.device)

        edge_features_1, edge_features_2 = [], []
        for n in data['edge_labels_1']:
            if self.global_edge_labels.get(n) is None:
                tar_feature_1 = [0.0] * len(self.global_edge_labels)
                tar_feature_1[self.global_edge_labels[("Z-Others", "Z-Others")]] = 1.0
                edge_features_1.append(tar_feature_1)
            else:
                edge_features_1.append(
                    [1.0 if self.global_edge_labels[n] == i else 0.0 for i in self.global_edge_labels.values()])
        for n in data['edge_labels_2']:
            if self.global_edge_labels.get(n) is None:
                tar_feature_2 = [0.0] * len(self.global_edge_labels)
                tar_feature_2[self.global_edge_labels[("Z-Others", "Z-Others")]] = 1.0
                edge_features_2.append(tar_feature_2)
            else:
                edge_features_2.append(
                    [1.0 if self.global_edge_labels[n] == i else 0.0 for i in self.global_edge_labels.values()])

        new_data["edge_features_1"] = torch.FloatTensor(np.array(edge_features_1)).to(self.args.device)
        new_data["edge_features_2"] = torch.FloatTensor(np.array(edge_features_2)).to(self.args.device)

        norm_ged = data["ged"] / (0.5 * (len(data["node_labels_1"]) + len(data["node_labels_2"])))

        new_data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float().to(self.args.device)
        return new_data
