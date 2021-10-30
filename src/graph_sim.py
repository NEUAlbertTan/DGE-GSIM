import torch
import numpy as np
import torch.nn.functional as tnfunc
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from dataset import Dataset
from layers import AvePoolingModule, AttentionModule, TenorNetworkModule, NodeGraphMatchingModule, MT_NEGCN
from utils import calculate_loss


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

        """
        init paths
        """
        self.args.dataset_path = self.args.dataset_root_path + self.args.current_dataset_name + \
                                 "/" + self.args.current_dataset_name + "_dataset.pkl"
        self.args.training_root_path = self.args.dataset_root_path + self.args.current_dataset_name + "/train/"
        self.args.test_root_path = self.args.dataset_root_path + self.args.current_dataset_name + "/test/"
        self.args.ged_path = self.args.dataset_root_path + self.args.current_dataset_name + \
                             "/" + self.args.current_dataset_name + "_ged.pkl"
        self.args.save_path = "../pretrained_models/" + self.args.filename
        self.args.best_model_path = "../pretrained_models/" + self.args.filename + "-best-val"
        self.args.load_path = "../pretrained_models/" + self.args.filename

        self.dataset = Dataset(args)
        self.model = GraphSim(self.args, self.dataset.number_of_node_labels, self.dataset.number_of_edge_labels) \
            .to(self.args.device)

    def create_batches(self):
        batches = []
        for graph_pair_index in range(0, len(self.dataset.training_graph_index_pairs), self.args.batch_size):
            batches.append(
                self.dataset.training_graph_index_pairs[graph_pair_index: graph_pair_index + self.args.batch_size])
        return batches

    def process_batch(self, batch):
        self.optimizer.zero_grad()
        losses = 0
        for graph_index_pair in batch:
            data = self.dataset.get_data(graph_index_pair, mode="training")
            data = self.dataset.transfer_to_torch(data)
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
        for graph_index_pair in tqdm(self.dataset.validation_graph_index_pairs):
            data = self.dataset.get_data(graph_index_pair, mode="validation")
            ground_truth.append(calculate_normalized_ged(data))
            data = self.dataset.transfer_to_torch(data)
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

        self.scores = np.zeros(len(self.dataset.test_graph_index_pairs))
        self.ground_truth = np.zeros(len(self.dataset.test_graph_index_pairs))
        self.prediction_list = np.zeros(len(self.dataset.test_graph_index_pairs))
        prec_at_10_list = []
        prec_at_20_list = []
        tau_list = []
        rho_list = []
        batch_ground_truth = []
        batch_prediction_list = []

        for index, test_index_pair in tqdm(enumerate(self.dataset.test_graph_index_pairs)):
            data = self.dataset.get_data(test_index_pair, mode="test")
            data = self.dataset.transfer_to_torch(data)
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
