import os
from pathlib import Path
import argparse
import random
from copy import deepcopy
import time

import torch
from torch import nn
import torch.utils.data as Data

import einops

from utils.kvb_core import BottleneckedEncoder
from backbone.model import GCN, GAT, MLP
from utils.split_data import *
from utils.model_utils import *
from utils.base import *
from utils.optimizer import get_optimizer


###################################### argsparse ######################################
parser = argparse.ArgumentParser(description='Training for Graph Few-Shot Incremental Learning')
parser.add_argument('--scaling_mode', type=str, default="free_num_keys", help='free_num_keys only so far')
parser.add_argument('--num_channels', type=int, default=16, help='num_channels')
parser.add_argument('--num_pairs', type=int, default=800, help='num_pairs')
parser.add_argument('--num_books', type=int, default=1, help='discrete codes num books')
parser.add_argument('--t_mode', type=str, default="uniform_importance", help='uniform_importance only mode so far')
parser.add_argument('--threshold_factor', type=float, default=0.1, help='threshold_factor')
parser.add_argument('--batch_size', type=int, default=256, help='Number of tasks in a mini-batch of tasks (default: 256).')
parser.add_argument('--dim_key', type=int, default=14, help='dim_key')
parser.add_argument('--dim_value', type=int, default=10, help='if dim_value is 0, then it is the same as dim_key')
parser.add_argument('--values_init', type=str, default="zeros", help='rand or zeros')
parser.add_argument('--cl_epochs', type=int, default=2000, help='cl_epochs')
parser.add_argument('--log_step_size', type=int, default=100)
parser.add_argument('--gradient_clip', type=float, default=1.0)
parser.add_argument('--method', type=str, default="ours", help='ours or mlp')
parser.add_argument('--decoder_model', type=str, default="codebook-voting-logits", help='decoder_model: codebook-voting-logits or linear-probe or mlp-128 or lp-no-bias')
parser.add_argument('--add_distance_to_values',action='store_true', default=False, help='add_distance_to_values')
parser.add_argument('--ff_dropout', type=float, default=0.0, help='ff_dropout')


parser.add_argument("--backbone", type=str, default='GCN', help='GCN, GAT')
parser.add_argument("--dataset", type=str, default='CoraFull', help='ogbn-arxiv, Reddit, ogbn-products, CoraFull')
parser.add_argument("--hidden", type=int, default=16, help='Number of hidden units')
parser.add_argument("--dropout", type=float, default=0.5, help='Dropout rate (1 - keep probability)')
parser.add_argument('--sparse', action='store_true', default=False, help='use sparse version of GAT for large graphs')
parser.add_argument("--nb_heads", type=int, default=8, help='Number of head attentions')
parser.add_argument("--alpha", type=float, default=0.2, help='Alpha for the leaky_relu')
parser.add_argument("--seed", type=int, default='42', help='ogbn-arxiv, Reddit, ogbn-products, CoraFull')
parser.add_argument("--pretrain_model_path", type=str, default='pretrain_model/CoraFull/GCN/42_490.pth', help='pretrain model path')
parser.add_argument("--optimizer", type=str, default='SGD', help='way to optimize the model')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning_rate')

parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--split_size', type=int, default=2, help='split_size')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
parser.add_argument('--init_epochs', type=int, default=1, help='init_epochs')
parser.add_argument('--GFCIL', type=str, default='high_resources', help='Whether to store the adj of past tasks')

args = parser.parse_args()


def initialize_model(base_adj, pretrain_idx, features, labels, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device variable: ", str(device))

    key_value_pairs_per_codebook = get_key_value_pairs_per_codebook(args.scaling_mode, args.num_channels, args.num_pairs, args.num_books) # 800
    threshold_ema_dead_code = get_threshold_ema_dead_code(args.num_channels, args.t_mode, args.scaling_mode, args.num_pairs, 
                                args.num_books, args.threshold_factor, features.shape[0]) # float 0.032
    checkpoint = torch.load(args.pretrain_model_path)
    
    if args.backbone == 'GCN':
        encoder = GCN(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)
    elif args.backbone == 'GAT':
        encoder = GAT(nfeat=features.shape[1], 
                    nhid=args.hidden,
                    nlayers=2,
                    dropout=args.dropout, 
                    alpha=args.alpha,
                    nheads=args.nb_heads,
                    use_sparse=args.sparse
                    )
        
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    
    bottlenecked_encoder = BottleneckedEncoder(
                                    encoder,
                                    base_adj,
                                    encoder_is_channel_last=False,
                                    num_channels=args.num_channels,
                                    num_codebooks=args.num_books,
                                    key_value_pairs_per_codebook=key_value_pairs_per_codebook,
                                    splitting_mode="random_projection",
                                    dim_keys=args.dim_key,
                                    dim_values=args.dim_value,
                                    decay=0.95,
                                    eps=1e-5,
                                    threshold_ema_dead_code=threshold_ema_dead_code,
                                    return_values_only=False,)
    
    print("Initializing model")
    bottlenecked_encoder.to(device)
    pretrain_data = Data.TensorDataset(features, labels)
    pretrain_data_loader = Data.DataLoader(pretrain_data, batch_size=features.shape[0], shuffle = True)
    bottlenecked_encoder = bottlenecked_encoder.prepare(pretrain_idx, loader=pretrain_data_loader, epochs=args.init_epochs)

    bottlenecked_encoder.reset_cluster_size_counter()

    bottlenecked_encoder.disable_update_keys()

    decoder = get_decoder_module(
        num_codebooks=bottlenecked_encoder.num_codebooks,
        dim_values=bottlenecked_encoder.dim_values,
        dim_keys=bottlenecked_encoder.dim_keys,
        num_channels=bottlenecked_encoder.num_channels,
        dataset_name=args.dataset,
        add_distance_to_values=args.add_distance_to_values,
        decoder_model=args.decoder_model,
        method=args.method,
    )

    model = ModelWrapper(bottlenecked_encoder, decoder, method=args.method, ff_dropout=args.ff_dropout, decoder_model=args.decoder_model)
    model.to(device)
    model.train()

    return model


def train(adj, base_adj, base_id, pretrain_idx, features, labels, id_by_class, novel_id, base_test_id, args):
    seed_everything(args.seed)
    print("Training run configs: " , str(args))
    print('base id: ', base_id)

    model = initialize_model(base_adj, pretrain_idx, features, labels, args)
    mlp = MLP(128, 70).cuda()
    
    if args.values_init == "randn":
        values = torch.randn_like(model.bottlenecked_encoder.bottleneck.values)
    elif args.values_init == "rand":
        values = torch.rand_like(model.bottlenecked_encoder.bottleneck.values)
    elif args.values_init == "zeros":
        values = torch.zeros_like(model.bottlenecked_encoder.bottleneck.values)
    else:
        raise ValueError("Unknown values_init: " + args.values_init)
    
    model.bottlenecked_encoder.bottleneck.values = nn.Parameter(values)
    model.bottlenecked_encoder.bottleneck.values.requires_grad = True
    model.bottlenecked_encoder.synapse.requires_grad = True
    model.bottlenecked_encoder.bottleneck.equires_grad = True
    
    
    random.shuffle(novel_id)

    class_splits = []
    split_size = args.split_size
    epoch_factor = split_size / 2
    num_splits = len(novel_id) // split_size
    for i in range(num_splits):
        class_splits.append(novel_id[i * split_size: (i + 1) * split_size])
    print("class_splits: " ,str(class_splits))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = get_optimizer(model, args)
    opt_mlp = get_optimizer(mlp, args)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    CE = DistillationLoss(temperature=1)

    model.bottlenecked_encoder.freeze_keys()
    model.freeze_for_adaptation()
    print("model is on device: ", str(next(model.parameters()).device))
    print("device variable: ", str(device))
    log_step_size = args.log_step_size
    loss_total = 0.0
    train_epochs = 0
    epoch = 0

    model.train()
    mlp.train()
    
    model.bottlenecked_encoder.activate_counts()
    cl_test = []
    cl_train = []
    cl_val = []
    base_test_id = [tensor.item() for tensor in base_test_id]
    class_splits.insert(0, list(set(base_test_id)))
   
    for class_list in class_splits:
        novel_train_idx, novel_val_idx, novel_test_idx, all_novel_idx = split_novel_data(class_list, id_by_class, args.dataset)
        print('class_list', class_list, "|", 'novel_train_idx is:', novel_train_idx)
        cl_test.append(novel_test_idx)
        cl_train.extend(novel_train_idx)
        cl_val.extend(novel_val_idx)
        
        min_acc = 0.0
       
        for epoch in range(0, args.cl_epochs):
            train_epochs += epoch_factor
            if epoch % log_step_size == 1:
                print("local_loss", loss_total)
                print("norm_values", torch.norm(
                            model.bottlenecked_encoder.bottleneck.values, dim=-1
                        ).mean())
                model.train()
                mlp.train()
                # mlp_1.train()
                model.bottlenecked_encoder.activate_counts()
                print("cl epoch: ", str(epoch))
            loss_total = 0.0
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            opt_mlp.zero_grad()
            
            base_id = base_id + class_list
            incremental_adj = get_incremental_adj_high(adj, base_id, all_novel_idx, labels)
            train_outputs, train_key, train_proj, tr_key, train_a = model(features, incremental_adj, cl_train, novel_train_idx)
            key_output = mlp(train_key)
            

            train_mlp_loss = criterion(key_output, labels[cl_train])
            train_kv_loss = criterion(train_outputs, labels[cl_train])
            train_loss = 0.5 * CE(train_outputs, key_output) + 0.5 * CE(key_output, train_outputs) + train_mlp_loss
            # train_loss = train_mlp_loss + train_loss
            # train_loss = criterion(train_outputs, labels[cl_train]) + 0.1 * torch.norm(proj_dist - train_dist, p=2)
            train_acc = accuracy(train_outputs, labels[cl_train])
            train_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            opt_mlp.step()
            
            loss_total += float(train_loss.item())

            model.eval()
            mlp.eval()
            
            val_outputs, val_key, val_proj, _, __ = model(features, incremental_adj, cl_val, novel_val_idx)
            val_keys = mlp(val_key)
            
            kv_val_loss = criterion(val_outputs, labels[cl_val])
            val_mlp_loss = criterion(val_keys, labels[cl_val])
            val_loss = 0.5 * CE(val_outputs, val_keys)+ 0.5 * CE(val_keys, val_outputs) + val_mlp_loss
            val_acc = accuracy(val_outputs, labels[cl_val])

            print('Epoch:{:04d}'.format(epoch),
                'train',
                'loss:{:.3f}'.format(train_loss),
                'acc:{:.2f}'.format(train_acc*100),
                '| val',
                'loss:{:.3f}'.format(val_loss),
                'acc:{:.2f}'.format(val_acc*100))
            
            
            if min_acc < val_acc:
                min_acc = val_acc
                torch.save(model.state_dict(), checkpt_file)
            torch.cuda.empty_cache()
        
        model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        mlp.eval()
        # mlp_1.eval()
        with torch.no_grad():
            test_cl = [item for sublist in cl_test for item in sublist]
            test_outputs, test_key, test_proj, chosen_key, a = model(features, incremental_adj, test_cl, cl_test)
            test_keys = mlp(test_key)
            
            test_kv_loss = criterion(test_outputs, labels[test_cl])
            test_mlp_loss = criterion(test_keys, labels[test_cl])
            test_loss = 0.5*CE(test_outputs, test_keys)+ 0.5*CE(test_keys, test_outputs) + test_mlp_loss
            
            for i in cl_test:
                test, _, __, ___, ____ = model(features, incremental_adj, i, i)
                print(accuracy(test, labels[i]))

            test_acc = accuracy(test_outputs, labels[test_cl])
            print('session {}:'.format(len(cl_test)),
                'test_loss:{:.2f}'.format(test_loss), 
                'session_acc:{:.2f}'.format(test_acc*100))

                

if __name__ == '__main__':
    dataset = args.dataset
    cache_path = os.path.join("./cache", str(dataset) + ".pth")
    cache = load_object(cache_path)

    pretrain_seed = cache["pretrain_seed"]
    adj = cache["adj"].cuda()
    base_adj = cache["pretrain_adj"].cuda()
    features = cache["features"]
    features = torch.tensor(features).cuda()
    labels = cache["labels"].cuda()
    id_by_class = cache["id_by_class"]
    novel_id = cache["novel_id"]
    base_id = cache["base_id"]
    base_test_id = cache["base_test_id"]
    pretrain_idx = cache["pretrain_idx"]
    del(cache)

    
    current_time = time.strftime("%d_%H_%M_%S", time.localtime(time.time()))
    checkpt_file = 'models/'+"{}_{}_{}".format(args.backbone, args.dataset, current_time)+'.pt'
    print(current_time,checkpt_file)


    train(adj, base_adj, base_id, pretrain_idx, features, labels, id_by_class, novel_id, base_test_id, args)








