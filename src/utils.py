from texttable import Texttable
import numpy as np


def get_file_id_from_path(path):
    import re

    pattern = r"(\\|/)[\w]*.gexf"
    re_res = re.search(pattern, path)

    # re_res.regs[0] represents a tuple, which indicates the begin and end index
    # of the whole matching substring, thus we plus begin_index by 1 to skip the '/'
    # and minus end_index by 5 to skip the '.gexf', to get the file name.
    begin, end = re_res.regs[0][0] + 1, re_res.regs[0][1] - 5

    return path[begin: end]


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transformed GED.
    :return score: Squared error.
    """
    # prediction = -math.log(prediction)
    # target = -math.log(target)
    score = (prediction-target)**2
    return score.item()


def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"]/(0.5*(len(data["node_labels_1"])+len(data["node_labels_2"])))
    return norm_ged


def get_data_from_path_pair(path_pair, ged_dic):
    import networkx as nx

    def get_file_id_from_path(path):
        import re

        pattern = r"(\\|/)[\w]*.gexf"
        re_res = re.search(pattern, path)

        # re_res.regs[0] represents a tuple, which indicates the begin and end index
        # of the whole matching substring, thus we plus begin_index by 1 to skip the '/'
        # and minus end_index by 5 to skip the '.gexf', to get the file name.
        begin, end = re_res.regs[0][0] + 1, re_res.regs[0][1] - 5

        return path[begin: end]

    data = dict()

    g1 = nx.read_gexf(path_pair[0])
    g2 = nx.read_gexf(path_pair[1])

    # process node labels
    node_labels_1, node_labels_2 = [], []

    for node in g1.nodes.data():
        node_labels_1.append(node[1]['label'])
    for node in g2.nodes.data():
        node_labels_2.append(node[1]['label'])

    data["labels_1"] = node_labels_1
    data["labels_2"] = node_labels_2

    graph_1, graph_2 = [], []
    for edges in g1.edges():
        graph_1.append(list(edges))
    for edges in g2.edges():
        graph_2.append(list(edges))

    data['graph_1'] = graph_1
    data['graph_2'] = graph_2

    # get GED
    g1_id = get_file_id_from_path(path_pair[0])
    g2_id = get_file_id_from_path(path_pair[1])

    data['ged'] = ged_dic.get((g1_id, g2_id))

    return data


def get_data_from_path(path, dataset_name="LINUX"):
    import networkx as nx

    data = dict()

    g1 = nx.read_gexf(path)

    # process node labels
    node_labels_1 = []

    node_index = {val: index for index, val in enumerate(g1.nodes)}

    for node in g1.nodes.data():
        if dataset_name == "AIDS700nef":
            node_labels_1.append(node[1]['type'])
        else:
            node_labels_1.append(node[1]['label'])

    data["labels"] = node_labels_1

    graph = []
    for edges in g1.edges():
        graph.append([node_index[edges[0]], node_index[edges[1]]])

    data['graph'] = graph

    data['id'] = int(get_file_id_from_path(path))

    return data


def ranking_func(data):
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    rank = np.zeros(n)
    for i in range(n):
        finds = np.where(sort_id_mat == i)
        fid = finds[0][0]
        while fid > 0:
            cid = sort_id_mat[fid]
            pid = sort_id_mat[fid - 1]
            if data[pid] == data[cid]:
                fid -= 1
            else:
                break
        rank[i] = fid + 1

    return rank


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    r_prediction = ranking_func(prediction)
    r_target = ranking_func(target)

    return rank_corr_function(r_prediction, r_target).correlation


def top_k_ids(data, k, inclusive, rm):
    """
    :param data: input
    :param k:
    :param inclusive: whether to be tie inclusive or not.
        For example, the ranking may look like this:
        7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
        If tie inclusive, the top 1 results are [7, 9].
        Therefore, the number of returned results may be larger than k.
        In summary,
            len(rtn) == k if not tie inclusive;
            len(rtn) >= k if tie inclusive.
    :param rm: 0
    :return: for a query, the ids of the top k database graph
    ranked by this model.
    """
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    if k < 0 or k >= n:
        raise RuntimeError('Invalid k {}'.format(k))
    if not inclusive:
        return sort_id_mat[:k]
    # Tie inclusive.
    dist_sim_mat = data
    while k < n:
        cid = sort_id_mat[k - 1]
        nid = sort_id_mat[k]
        if abs(dist_sim_mat[cid] - dist_sim_mat[nid]) <= rm:
            k += 1
        else:
            break
    return sort_id_mat[:k]


def prec_at_ks(true_r, pred_r, ks, rm=0):
    """
    Ranking-based. prec@ks.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param ks: k
    :param rm: 0
    :return: precision at ks.
    """
    true_ids = top_k_ids(true_r, ks, inclusive=True, rm=rm)
    pred_ids = top_k_ids(pred_r, ks, inclusive=True, rm=rm)
    ps = min(len(set(true_ids).intersection(set(pred_ids))), ks) / ks
    return ps

