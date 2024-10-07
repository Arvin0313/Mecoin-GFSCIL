import numpy as np
import torch
import scipy.sparse as sp
from sklearn import preprocessing
from torch_geometric.datasets import CoraFull, DBLP, NELL, Coauthor, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
import collections
import heapq
import random
import math
import pickle
import json
from scipy.sparse.csgraph import shortest_path


pretrain_cls_num = {'CoraFull':50, 'ogbn-arxiv':25, 'Reddit': 24, 'DBLP':87, 'Nell':166, 'CS':5, 'Computers': 5}
test_cls_num = {'CoraFull':27, 
                'ogbn-arxiv':16, 
                'Reddit': 16, 
                'DBLP': 50,
                'Nell':20,
                'CS':10,
                'Computers': 5}

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_raw_data(dataset):
    '''
    adj: sparse tensor, normalize_adj, (n,n)
    features: tensor, node features, (n, m)
    data.y: tensor, labels, (n)
    id_by_class: dict, key is class, value is the node index belonging to the key class
    base_id: list, class in pre-train
    novel_id: list, class in cl
    num_nodes: list, number of node in each class
    nodes_num: int,  number of node in whole graph
    '''
    base_num = pretrain_cls_num[dataset]
    if dataset == 'CoraFull':
        dataset = CoraFull(root='/home/bqqi/graph-lifelong-learning/graph-lifelong-learning/graph_kvb/dataset')
    elif dataset == 'DBLP':
        dataset = DBLP(root='/home/bqqi/graph-lifelong-learning/graph-lifelong-learning/graph_kvb/dataset')
    elif dataset == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/bqqi/graph-lifelong-learning/graph-lifelong-learning/graph_kvb/dataset')
    elif dataset == 'CS':
        dataset = Coauthor(root='/home/bqqi/graph-lifelong-learning/graph-lifelong-learning/graph_kvb/dataset', name='CS')
    elif dataset == 'Computers':
        dataset = Amazon(root='/home/bqqi/graph-lifelong-learning/graph-lifelong-learning/graph_kvb/dataset', name='Computers')
    else:
        raise ValueError('Unknown dataset')

    data = dataset[0] # data = (x=[19793, 8710], edge_index=[2, 126842], y=[19793])
    nodes_num = data.edge_index.max().item() + 1 # node num

    # get adj
    adj = torch.zeros(nodes_num, nodes_num)
    adj[data.edge_index[0], data.edge_index[1]] = 1
    adj[data.edge_index[1], data.edge_index[0]] = 1

    adj = adj.numpy()
    # shortest_paths, _ = shortest_path(adj, directed=False, return_predecessors=True)
    deg = np.diag(np.sum(adj, axis=1))
    deg = np.diag(deg)
    deg = torch.tensor(deg)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).coalesce()

    features = data.x
    id_by_class = {}
    class_list = []
    for cla in data.y.tolist():
        if cla not in class_list:
            class_list.append(cla)  # unsorted

    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(data.y.tolist()):
        id_by_class[cla].append(id)

    num_nodes = []
    for _, v in id_by_class.items():
        num_nodes.append(len(v))

    # 遍历data.y，将节点索引添加到相应的列表中
    for i, class_num in enumerate(data.y):
        id_by_class[class_num.item()].append(i)

    large_res_idex = heapq.nlargest(base_num, enumerate(num_nodes), key=lambda x: x[1])
    base_id = [id_num_tuple[0] for id_num_tuple in large_res_idex]
    all_id = [i for i in range(len(num_nodes))]
    novel_id = list(set(all_id).difference(set(base_id)))
    
    return adj, features, data.y, id_by_class, base_id, novel_id, num_nodes, nodes_num, deg

def split_base_data(base_id, id_by_class, labels):
    '''
    split pre-train data.
    input:
        base_id: list, class in pre-train
        id_by_class: dict, key is class, value is the node index belonging to the key class
        labels: tensor, labels, (n)
    output:
        pretrain_idx: list, training data index in pre-train
        preval_idx: list, val data index in pre-train
        pretest_idx: list, test data in pre-train
        base_train_label: tensor, each node of training data label in pre-train
        base_val_label: tensor, each node of val data label in pre-train
        base_test_label: tensor, each node of test data label in pretrain
        base_train_id: training data label in pre-train
        base_val_id: val data label in pre-train
        base_test_id: test data label in pretrain
    '''
    pretrain_idx = []
    preval_idx = []
    pretest_idx = []
    random.shuffle(base_id)
    print('base_id', sorted(base_id))
    print('id_by_class', sorted(list(id_by_class.keys())))

    for cla in base_id:
        node_idx = id_by_class[cla]
        num_nodes = len(node_idx)
        train_num = math.ceil(0.6 * num_nodes)
        dev_num = math.ceil(0.2 * num_nodes)
        test_num = num_nodes - train_num - dev_num

        random.shuffle(node_idx)
        pretrain_idx.extend(node_idx[: train_num])
        preval_idx.extend(node_idx[train_num: train_num + dev_num])
        pretest_idx.extend(node_idx[train_num + dev_num: train_num + dev_num + test_num])

    base_train_label = labels[pretrain_idx]
    base_val_label = labels[preval_idx]
    base_test_label = labels[pretest_idx]

    base_train_id = sorted(set(base_train_label))
    base_val_id = sorted(set(base_val_label))
    base_test_id = sorted(set(base_test_label))

    return pretrain_idx, preval_idx, pretest_idx, base_train_label, base_val_label, \
           base_test_label, base_train_id, base_val_id, base_test_id


def get_base_adj(adj, pretrain_idx, labels):
    '''
    get adj in pre-train.
    input:
        adj: sparse tensor, (n,n)
        pretrain_idx: list, training data index in pre-train
        label: tensor, labels, (n)
    output:
        base_adj: sparse tensor, (n,n)
    '''
    # I = adj.indices()
    # V = adj.values()
    I = adj.coalesce().indices()
    V = adj.coalesce().values()
    dim_base = len(labels)

    mask = []
    # for i in range(I.shape[1]):
    #     if labels[I[0, i]] in pretrain_idx and labels[I[1, i]] in pretrain_idx:
    #         mask.append(True)
    #     else:
    #         mask.append(False)
    mask = np.isin(labels[I[0]], pretrain_idx) & np.isin(labels[I[1]], pretrain_idx)
    mask = torch.tensor(mask)

    I_base = I[:, mask]
    V_base = V[mask]

    base_adj = torch.sparse_coo_tensor(I_base, V_base, (dim_base, dim_base)).coalesce()

    return base_adj


# def get_incremental_adj_high(adj, base_id, novel_idx, labels):
#     I = adj.indices()
#     V = adj.values()
#     dim_base = len(labels)
#     # novel_idx = np.append(novel_id_support, novel_id_query)

#     mask = []
#     for i in range(I.shape[1]):
#         if (labels[I[0, i]] in base_id and labels[I[1, i]] in base_id) or \
#                 (I[0, i] in novel_idx and I[1, i] in novel_idx):
#             mask.append(True)
#         else:
#             mask.append(False)
#     mask = torch.tensor(mask)
#     I_incremental = I[:, mask]
#     V_incremental = V[mask]

#     incremental_adj = torch.sparse_coo_tensor(I_incremental, V_incremental, (dim_base, dim_base)).coalesce()

#     return incremental_adj
def get_incremental_adj_high(adj, base_id, novel_idx, labels):
    adj = adj.coalesce()
    I = adj.indices().cpu()
    V = adj.values().cpu()
    dim_base = len(labels)

    adj = adj.cpu()
    labels = labels.cpu()
    
    # Check membership using NumPy operations
    in_base_id = np.isin(labels[I[0]], base_id) & np.isin(labels[I[1]], base_id)
    in_novel_idx = np.isin(I[0], novel_idx) & np.isin(I[1], novel_idx)
    
    mask = in_base_id | in_novel_idx
    # mask = in_novel_idx
    
    # Use PyTorch operations to filter indices and values
    I_incremental = I[:, mask]
    V_incremental = V[mask]
    
    # Create sparse tensor
    incremental_adj = torch.sparse_coo_tensor(I_incremental, V_incremental, (dim_base, dim_base)).coalesce()
    incremental_adj = incremental_adj.cuda()

    return incremental_adj


def get_incremental_adj_low(adj, base_id, novel_id_support, novel_id_query, labels):
    I = adj.indices()
    V = adj.values()
    dim_base = len(labels)
    novel_idx = np.append(novel_id_support, novel_id_query)

    mask = []
    for i in range(I.shape[1]):
        if (labels[I[0, i]] in base_id and labels[I[1, i]] in base_id) or \
                (I[0, i] in novel_idx and I[1, i] in novel_idx):
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)
    I_incremental = I[:, mask]
    V_incremental = V[mask]

    incremental_adj = torch.sparse_coo_tensor(I_incremental, V_incremental, (dim_base, dim_base)).coalesce()

    return incremental_adj


def incremental_task_generator(id_by_class, n_way, k_shot, m_query, base_id, novel_id):
    '''
    input:
        id_by_class: dict, key is class, value is the node index belonging to the key class
        n_way: int
        k_shot: int, k shot for query set
    output:
        np.array(base_id_query): array, node index in pre-train training data
        np.array(novel_id_query): array, node index in cl for testing
        np.array(novel_id_support): array, node index in cl for training
        base_class_selected: base_id
        novel_class_selected: class in cl for task i 
    '''
    # sample class indices
    base_class_selected = base_id
    novel_class_selected = random.sample(novel_id, n_way)

    novel_id_support = []
    novel_id_query = []
    base_id_query = []
    for cla in novel_class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        novel_id_support.extend(temp[:k_shot])
        novel_id_query.extend(temp[k_shot:])
    for cla in base_class_selected:
        temp = random.sample(id_by_class[cla], m_query)
        base_id_query.extend(temp)
    return np.array(base_id_query), np.array(novel_id_query), np.array(novel_id_support), \
           base_class_selected, novel_class_selected


def get_incremental_adj(adj, base_id, novel_id_support, novel_id_query, labels):
    '''
    output:
        incremental_adj: The adjacency matrix composed of the current task and pre-training nodes
    '''
    I = adj.indices()
    V = adj.values()
    dim_base = len(labels)
    novel_idx = np.append(novel_id_support, novel_id_query)

    mask = []
    for i in range(I.shape[1]):
        if (labels[I[0, i]] in base_id and labels[I[1, i]] in base_id) or \
                (I[0, i] in novel_idx and I[1, i] in novel_idx):
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)
    I_incremental = I[:, mask]
    V_incremental = V[mask]

    incremental_adj = torch.sparse_coo_tensor(I_incremental, V_incremental, (dim_base, dim_base)).coalesce()

    return incremental_adj


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# def save_object(obj, filename):
#     with open(filename, 'wb') as fout:  # Overwrites any existing file.
#         pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)


# def load_object(filename):
#     with open(filename, 'rb') as fin:
#         obj = pickle.load(fin)
#     return obj
def save_object(obj, filename):
    # with open(filename, 'w') as fout:
    #     json.dump(obj, fout)
    torch.save(obj, filename)

def load_object(filename):
    # with open(filename, 'r') as fin:
    #     return json.load(fin)
    return torch.load(filename)
    


def split_novel_data(novel_id, id_by_class, dataset_source):
    split_dict = test_cls_num[dataset_source]
    random.shuffle(novel_id)
    novel_train_idx = []
    novel_test_idx = []
    novel_val_idx = []
    all_novel_idx = []
    for i in novel_id:
        # novel_train_idx.extend(id_by_class[i][:int(len(id_by_class[i]) * 0.6)])
        # novel_val_idx.extend(id_by_class[i][int(len(id_by_class[i]) * 0.6):int(len(id_by_class[i]) * 0.8)])
        # novel_test_idx.extend(id_by_class[i][int(len(id_by_class[i]) * 0.8):])
        novel_train_idx.extend(id_by_class[i][:5])
        novel_val_idx.extend(id_by_class[i][5:int(len(id_by_class[i]) * 0.2)])
        novel_test_idx.extend(id_by_class[i][int(len(id_by_class[i]) * 0.2):])
        all_novel_idx.extend(id_by_class[i])
        

    return novel_train_idx, novel_val_idx, novel_test_idx, all_novel_idx