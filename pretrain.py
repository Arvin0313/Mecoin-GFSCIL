import time
import argparse
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.split_data import *
from backbone.model import GCN, GAT, MLP
from utils.argparse import *
from backbone.classifier import Classifier

###################################### argsparse ######################################
parser = argparse.ArgumentParser(description='Pre-train for Graph Few-Shot Incremental Learning')
parser.add_argument("--dataset", type=str, default='CoraFull', help='ogbn-arxiv, Reddit, ogbn-products, CoraFull')
parser.add_argument("--backbone", type=str, default='GCN', help='GCN, GAT')
parser.add_argument("--nb_heads", type=int, default=8, help='Number of head attentions')
parser.add_argument("--alpha", type=float, default=0.2, help='Alpha for the leaky_relu')
parser.add_argument("--lr", type=float, default=0.005, help='Initial learning rate')
parser.add_argument("--weight_decay", type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--sparse', action='store_true', default=False, help='use sparse version of GAT for large graphs')

parser.add_argument("--seed", type=int, default='42', help='ogbn-arxiv, Reddit, ogbn-products, CoraFull, Nell, CS')
parser.add_argument("--output_path", default='./pretrain_model', help='Path for output pre-trained model.')
parser.add_argument('--overwrite_pretrain', action='store_true', help='Delete existing pre-train model')

parser.add_argument("--epochs", type=int, default=2000, help='Number of epochs to train')
parser.add_argument("--lazy", type=int, default=10, help='Lazy epoch to terminate pre-training')
parser.add_argument("--hidden", type=int, default=16, help='Number of hidden units')
parser.add_argument("--dropout", type=float, default=0.5, help='Dropout rate (1 - keep probability)')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mkdir for pre-train model
path_tmp = os.path.join(args.output_path, str(args.dataset), args.backbone)
if args.overwrite_pretrain and os.path.exists(path_tmp):
    cmd = "rm -rf " + path_tmp
    os.system(cmd)

if not os.path.exists(path_tmp):
    os.makedirs(path_tmp)

# set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)


# load dataset
if args.dataset == 'Reddit':
    from utils.process_reddit import process_reddit
    adj, features, labels, id_by_class, base_id, novel_id, num_nodes, num_all_nodes = process_reddit(args.dataset)
elif args.dataset == 'ogbn-arxiv':
    from utils.process_arxiv import process_arxiv
    adj, features, labels, id_by_class, base_id, novel_id, num_nodes, num_all_nodes = process_arxiv(args.dataset)
else:
    adj, features, labels, id_by_class, base_id, novel_id, num_nodes, num_all_nodes, deg = load_raw_data(args.dataset)
pretrain_idx, preval_idx, pretest_idx, base_train_label, base_val_label, \
           base_test_label, base_train_id, base_val_id, base_test_id = split_base_data(base_id, id_by_class, labels)
pretrain_adj = get_base_adj(adj, base_id, labels)


# save the pre-train graph
cache = {"pretrain_seed": args.seed, "adj": adj, "features": features, "labels": labels, "pretrain_adj": pretrain_adj,
         "pretrain_idx":pretrain_idx, "id_by_class": id_by_class, "base_id": base_id,
         "novel_id": novel_id, "num_nodes": num_nodes, "num_all_nodes": num_all_nodes,
         "base_train_id": base_train_id, "base_dev_id": base_val_id, "base_test_id": base_test_id, "pretest_idx": pretest_idx,
         "preval_idx": preval_idx, "deg": deg}
cache_path = os.path.join("./cache", (str(args.dataset) + ".pth"))
if not os.path.exists("./cache"):
    os.makedirs("./cache")
save_object(cache, cache_path)
del cache



# pre-train model and optimizer
if args.backbone == 'GCN':
    model = GCN(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)
elif args.backbone == 'GAT':
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden,
                nlayers=2,
                dropout=args.dropout, 
                alpha=args.alpha,
                nheads=args.nb_heads,
                use_sparse=args.sparse
                )

classifier = Classifier(nhid=args.hidden, nclass=int(labels.max()) + 1)


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_classifier = optim.Adam(classifier.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# put data into device
model.to(device)
classifier.to(device)
features = torch.tensor(features)
print(features.shape)
features = features.to(device)
pretrain_adj = pretrain_adj.to(device)
labels = labels.to(device)

def pretrain_epoch(pretrain_idx, features):
    model.train()
    classifier.train()
    optimizer.zero_grad()
    embeddings = model(features, pretrain_adj).to(device)
    output = classifier(embeddings)[pretrain_idx]
    output = F.log_softmax(output, dim=1).to(device)

    base_train_label = torch.LongTensor([i for i in labels[pretrain_idx]]).to(device)
    loss_train = F.nll_loss(output, base_train_label)
    loss_train.backward()
    optimizer.step()
    optimizer_classifier.step()

    output = output.cpu().detach()
    base_train_label = base_train_label.cpu().detach()
    acc_train = accuracy(output, base_train_label)

    return acc_train

def pretest_epoch(pretest_idx, features):
    model.eval()
    classifier.eval()
    embeddings = model(features, pretrain_adj)
    output = classifier(embeddings)[pretest_idx]
    output = F.log_softmax(output, dim=1)

    base_test_label = torch.LongTensor([i for i in labels[pretest_idx]])
    base_test_label = base_test_label.to(device)
    loss_test = F.nll_loss(output, base_test_label)

    output = output.cpu().detach()
    base_test_label = base_test_label.cpu().detach()
    acc_test = accuracy(output, base_test_label)

    return acc_test


if __name__ == '__main__':
    t_total = time.time()
    pre_train_acc = []

    best_dev_acc = 0.
    tolerate = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        acc_train = pretrain_epoch(pretrain_idx, features)
        pre_train_acc.append(acc_train)
        if epoch > 0 and epoch % 10 == 0:
            print("-------Epochs {}-------".format(epoch))
            print("Pre-Train_Accuracy: {}".format(np.array(pre_train_acc).mean(axis=0)))

            # validation
            pre_dev_acc = []

            acc_test = pretest_epoch(preval_idx, features)
            pre_dev_acc.append(acc_test)
            curr_dev_acc = np.array(pre_dev_acc).mean(axis=0)
            print("Pre-valid_Accuracy: {}".format(curr_dev_acc))
            if curr_dev_acc > best_dev_acc:
                best_dev_acc = curr_dev_acc
                save_path = os.path.join(args.output_path, args.dataset, args.backbone, str(args.seed) + "_" + (str(epoch) + ".pth"))
                tolerate = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'loss': loss,
                }, save_path)
                print("model saved at " + save_path)
                best_epoch = epoch
            else:
                tolerate += 1
                if tolerate > args.lazy:
                    print("Pretraining finished at epoch: " + str(epoch))
                    print("Best pretrain epoch: " + str(best_epoch))
                    break
            # testing
            pre_test_acc = []
            pre_test_f1 = []
            acc_test = pretest_epoch(pretest_idx, features)
            pre_test_acc.append(acc_test)
            print("Pre-Test_Accuracy: {}".format(np.array(pre_test_acc).mean(axis=0)))

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
