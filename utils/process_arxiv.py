# from ogb.nodeproppred import PygNodePropPredDataset
import dgl
import numpy as np
import scipy.sparse as sp
import torch
import heapq
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
import yaml



pretrain_cls_num = {'CoraFull':50, 'ogbn-arxiv':24, 'Reddit': 21, 'ogbn-products': 27}
test_cls_num = {'CoraFull':27, 
                'ogbn-arxiv':16, 
                'Reddit': 16, 
                'ogbn-products': 20}


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def process_arxiv(dataset):
    base_num = pretrain_cls_num[dataset]
    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='/home/bqqi/graph-lifelong-learning/graph-lifelong-learning/graph_kvb/dataset')
    g = dataset[0][0]
    g = dgl.add_reverse_edges(g)
    g = dgl.to_simple(g)

    A = g.adj_external(scipy_fmt='csr')

    deg = np.array(A.sum(axis=0)).flatten()
    D_ = sp.diags(deg ** -0.5)

    adj = D_.dot(A.dot(D_))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = g.ndata['feat']
    labels = dataset[0][1]
    print(labels.type)

    id_by_class = {}
    class_list = []
    labels = np.argmax(labels, axis=1).tolist()
    labels = torch.tensor(labels)
    for cla in labels.tolist():
        if cla not in class_list:
            class_list.append(cla)  # unsorted

    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels.tolist()):
        id_by_class[cla].append(id)

    num_nodes = []
    for _, v in id_by_class.items():
        num_nodes.append(len(v))

    # 遍历data.y，将节点索引添加到相应的列表中
    for i, class_num in enumerate(labels.numpy()):
        id_by_class[class_num.item()].append(i)

    large_res_idex = heapq.nlargest(base_num, enumerate(num_nodes), key=lambda x: x[1])
    base_id = [id_num_tuple[0] for id_num_tuple in large_res_idex]
    all_id = [i for i in range(len(num_nodes))]
    novel_id = list(set(all_id).difference(set(base_id)))
    nodes_num = adj.shape[0]
    
    return adj, features, labels, id_by_class, base_id, novel_id, num_nodes, nodes_num



