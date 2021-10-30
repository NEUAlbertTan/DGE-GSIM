import torch
import random
import numpy as np
import torch.nn.functional as tnfunc
from tqdm import tqdm, trange
from layers import AvePoolingModule, AttentionModule, TenorNetworkModule, NodeGraphMatchingModule, MT_NEGCN
from utils import calculate_loss
from scipy.stats import spearmanr, kendalltau


class GraphSim(torch.nn.Module):

    def __init__(self, args, number_of_node_labels, number_of_edge_labels):
        super(GraphSim, self).__init__()
        self.args = args

        self.number_node_labels = number_of_node_labels
        self.number_edge_labels = number_of_edge_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        tensor_layer_out = self.args.tensor_neurons * 2

        if self.args.histogram:
            self.feature_count = tensor_layer_out + self.args.bins
        else:
            self.feature_count = tensor_layer_out

        # node-graph-features
        if self.args.node_graph_matching:
            self.feature_count = self.feature_count + self.args.hidden_size * 4

    def setup_layers(self):
        self.calculate_bottleneck_features()

        self.convolution_0 = MT_NEGCN(self.args, self.number_node_labels, self.number_edge_labels)

        if self.args.attention_module:
            self.attention = AttentionModule(self.args).to(self.args.device)
            self.attention_edge = AttentionModule(self.args).to(self.args.device)
        else:
            self.avePooling = AvePoolingModule(self.args).to(self.args.device)

        self.tensor_network = TenorNetworkModule(self.args).to(self.args.device)

        if self.args.node_graph_matching:
            self.node_graph_matching = NodeGraphMatchingModule(self.args).to(self.args.device)

        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons).to(self.args.device)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1).to(self.args.device)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        scores = torch.mm(abstract_features_1, abstract_features_2).detach().to(self.args.device)
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins).to(self.args.device)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def forward(self, data):

        abstract_features_1, edge_features_1 = self.convolution_0(data["node_features_1"], data["edge_index_1"],
                                                                  data["edge_features_1"], data["trans_edge_index_1"])
        abstract_features_2, edge_features_2 = self.convolution_0(data["node_features_2"], data["edge_index_2"],
                                                                  data["edge_features_2"], data["trans_edge_index_2"])

        if self.args.histogram:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        if self.args.tensor_network:
            if self.args.attention_module:

                pooled_edge_features_1 = self.attention_edge(edge_features_1)
                pooled_edge_features_2 = self.attention_edge(edge_features_2)
                pooled_features_1 = self.attention(abstract_features_1)
                pooled_features_2 = self.attention(abstract_features_2)
            else:
                pooled_features_1 = self.avePooling(abstract_features_1)
                pooled_features_2 = self.avePooling(abstract_features_2)
                pooled_edge_features_1 = self.avePooling(edge_features_1)
                pooled_edge_features_2 = self.avePooling(edge_features_2)

            scores_node = self.tensor_network(pooled_features_1, pooled_features_2)
            scores_edge = self.tensor_network(pooled_edge_features_1, pooled_edge_features_2)
            scores = torch.t(torch.cat((scores_node, scores_edge), dim=0))

            if self.args.histogram:
                scores = torch.cat((scores, hist), dim=1).view(1, -1)
        else:
            scores = hist.view(1, -1)

        if self.args.node_graph_matching:
            # node-graph sub-network
            node_graph_score = self.node_graph_matching(abstract_features_1, abstract_features_2)
            scores = torch.cat((scores, node_graph_score), dim=1).view(1, -1)

        scores = tnfunc.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score


class GraphSimTrainer(object):
    def __init__(self, args):
        self.args = args
        self.init_path()
        self.process_dataset()
        self.init_labels()
        self.setup_model()

    def init_path(self):
        self.args.dataset_path = self.args.dataset_root_path + self.args.current_dataset_name + \
                                 "/" + self.args.current_dataset_name + "_dataset.pkl"

        self.args.training_root_path = self.args.dataset_root_path + self.args.current_dataset_name + "/train/"
        self.args.test_root_path = self.args.dataset_root_path + self.args.current_dataset_name + "/test/"
        self.args.ged_path = self.args.dataset_root_path + self.args.current_dataset_name + \
                             "/" + self.args.current_dataset_name + "_ged.pkl"

        self.args.save_path = "../pretrained_models/" + self.args.filename
        self.args.best_model_path = "../pretrained_models/" + self.args.filename + "-best-val"
        self.args.load_path = "../pretrained_models/" + self.args.filename

    def process_dataset(self):
        import glob
        import pickle
        from utils import get_data_from_path
        training_paths = glob.glob(self.args.training_root_path + "*.gexf")
        test_paths = glob.glob(self.args.test_root_path + "*.gexf")
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

        self.ged_dict = pickle.load(open(self.args.ged_path, "rb"))

        self.training_graph_index_pairs = []
        for i in range(len(self.training_graphs)):
            for j in range(len(self.training_graphs)):
                self.training_graph_index_pairs.append((i, j))

        self.test_graph_index_pairs = []
        for i in range(len(self.test_graphs)):
            for j in range(len(self.training_graphs)):
                self.test_graph_index_pairs.append((i, j))

        if self.args.validate:
            n_training_pairs = len(self.training_graph_index_pairs)
            self.validation_graph_index_pairs = self.training_graph_index_pairs[int(0.75 * n_training_pairs):]
            self.training_graph_index_pairs = self.training_graph_index_pairs[:int(0.75 * n_training_pairs)]

    def init_labels(self):
        tar_node_set = set()
        tar_edge_set = set()

        edge_count_dict = dict()
        total_edge_num = 0
        for graph in self.training_graphs + self.test_graphs:
            tar_node_set |= set(graph["labels"])
            total_edge_num += len(graph["graph"])
            for edge in graph["graph"]:
                cur_edge_label = tuple(sorted([graph["labels"][edge[0]], graph["labels"][edge[1]]]))
                tar_edge_set.add(cur_edge_label)
                # edge_count_dict[cur_edge_label] = 1 if not edge_count_dict.get(cur_edge_label) else edge_count_dict[cur_edge_label] + 1

        # edge_rank = sorted(edge_count_dict.items(), key=lambda k_v: (k_v[1], k_v[0]), reverse=True)
        # temp_edge_count = 0
        # cut_edge_count = total_edge_num * self.args.label_cut_ratio
        # for e in edge_rank:
        #     tar_edge_set.add(e[0])
        #     temp_edge_count += e[1]
        #     if temp_edge_count >= cut_edge_count:
        #         break
        # tar_edge_set.add(("Z-Others", "Z-Others"))

        self.global_node_labels = sorted(tar_node_set)
        self.global_edge_labels = sorted(tar_edge_set)
        self.global_node_labels = {val: index for index, val in enumerate(self.global_node_labels)}
        self.global_edge_labels = {val: index for index, val in enumerate(self.global_edge_labels)}

        self.number_of_node_labels = len(self.global_node_labels)
        self.number_of_edge_labels = len(self.global_edge_labels)

    def setup_model(self):
        self.model = GraphSim(self.args, self.number_of_node_labels, self.number_of_edge_labels).to(self.args.device)

    def create_batches(self):
        batches = []
        for graph_pair_index in range(0, len(self.training_graph_index_pairs), self.args.batch_size):
            batches.append(self.training_graph_index_pairs[graph_pair_index: graph_pair_index + self.args.batch_size])
        return batches

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

    def process_batch(self, batch):
        self.optimizer.zero_grad()
        losses = 0
        for graph_index_pair in batch:
            data = self.get_data(graph_index_pair, mode="training")
            data = self.transfer_to_torch(data)
            prediction = self.model(data)
            losses = losses + tnfunc.mse_loss(data["target"], prediction)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def validate(self, index):
        from utils import calculate_normalized_ged
        self.model.eval()
        print("\n\nModel evaluation.\n")
        scores = []
        ground_truth = []
        for graph_index_pair in tqdm(self.validation_graph_index_pairs):
            data = self.get_data(graph_index_pair, mode="validation")
            ground_truth.append(calculate_normalized_ged(data))
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            scores.append(calculate_loss(prediction, target))
        model_error = np.mean(scores)
        self.epoch_loss_list.append(model_error)
        print("\nModel validate error: " + str(round(float(model_error), 5)) + ".")
        if model_error < self.min_error:
            self.best_epoch_index = index
            self.min_error = model_error
            torch.save(self.model.state_dict(), self.args.best_model_path)

    def train(self):
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        self.model.to(self.args.device)

        self.min_error = 100
        self.best_model = None
        self.best_epoch_index = 0
        self.epoch_loss_list = []

        for epoch_index, epoch in enumerate(epochs):
            self.model.train()
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for batch in tqdm(batches, total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum / main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            if self.args.validate:
                self.validate(epoch_index)

        if self.args.validate:
            self.model.load_state_dict(torch.load(self.args.best_model_path))

    def test(self):
        from utils import calculate_ranking_correlation, prec_at_ks

        print("\n\nModel testing.\n")
        self.model.eval()

        self.scores = np.zeros(len(self.test_graph_index_pairs))
        self.ground_truth = np.zeros(len(self.test_graph_index_pairs))
        self.prediction_list = np.zeros(len(self.test_graph_index_pairs))
        prec_at_10_list = []
        prec_at_20_list = []
        tau_list = []
        rho_list = []
        batch_ground_truth = []
        batch_prediction_list = []

        for index, test_index_pair in tqdm(enumerate(self.test_graph_index_pairs)):
            data = self.get_data(test_index_pair, mode="test")
            data = self.transfer_to_torch(data)
            target = data["target"]
            self.ground_truth[index] = target
            prediction = self.model(data)
            self.prediction_list[index] = prediction
            self.scores[index] = calculate_loss(prediction, target)
            batch_ground_truth.append(target.item())
            batch_prediction_list.append(prediction.item())
            if (index + 1) % self.args.batch_size == 0:
                np_batch_gt = np.array(batch_ground_truth)
                np_batch_p = np.array(batch_prediction_list)
                prec_at_10_list.append(prec_at_ks(np_batch_gt, np_batch_p, 10))
                prec_at_20_list.append(prec_at_ks(np_batch_gt, np_batch_p, 20))
                rho_list.append(calculate_ranking_correlation(spearmanr, np_batch_p, np_batch_gt))
                tau_list.append(calculate_ranking_correlation(kendalltau, np_batch_p, np_batch_gt))
                batch_ground_truth.clear()
                batch_prediction_list.clear()

        mse = np.mean(self.scores)
        rho = np.mean(rho_list)
        tau = np.mean(tau_list)
        p_at_20 = np.mean(prec_at_20_list)
        p_at_10 = np.mean(prec_at_10_list)
        self.print_evaluation(mse, rho, tau, p_at_20, p_at_10)

    def print_evaluation(self, mse, rho, tau, prec_at_20, prec_at_10):
        mean_ground_truth = np.mean(self.ground_truth)
        mean_predicted = np.mean(self.prediction_list)
        delta = np.mean([(n - mean_ground_truth) ** 2 for n in self.ground_truth])
        predicted_delta = np.mean([(n - mean_predicted) ** 2 for n in self.prediction_list])

        print("\nGround truth delta: " + str(round(float(delta), 5)) + ".")
        print("\nPredicted delta: " + str(round(float(predicted_delta), 5)) + ".")
        print("\nModel test error(mse): " + str(round(float(mse), 5)) + ".")
        print("rho: ", rho)
        print("tau: ", tau)
        print("p@20:", prec_at_20)
        print("p@10:", prec_at_10)
        
        if self.args.validate:
            print("\nModel validation loss in each epoch:", self.epoch_loss_list)
            print("\nBest epoch index: " + str(self.best_epoch_index))
            print("\nBest epoch validate error: " + str(self.min_error))

    def save(self, path=""):
        if path == "":
            path = self.args.save_path
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))
