import scipy.sparse as sp
import os.path
import numpy as np
import heapq
import torch


file_dir = '/home/bqqi/graph-lifelong-learning/graph-lifelong-learning/graph_kvb/dataset/reddit'
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


def process_reddit(dataset):
    base_num = pretrain_cls_num[dataset]
    adj = sp.load_npz(os.path.join(file_dir, 'reddit_adj.npz'))
    features = np.load(os.path.join(file_dir, 'reddit_feat.npy'))
    labels = np.load(os.path.join(file_dir, 'reddit_labels.npy'))

    adj = adj + sp.eye(adj.shape[0])
    D1 = np.array(adj.sum(axis=1))**(-0.5)
    D2 = np.array(adj.sum(axis=0))**(-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)
    adj = A

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
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    return adj, features, labels, id_by_class, base_id, novel_id, num_nodes, nodes_num



