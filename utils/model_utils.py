import argparse

import einops
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from .base import get_class_nums
# from .kvb_core import values_interact_x


class CodebookVotingLogitsDecoder(nn.Module):
    def __init__(self, dim_values, class_nums, topk, ff_dropout=0.0):
        super().__init__()
        self.dim_values = dim_values
        self.class_nums = class_nums
        self.ff_dropout = 0.0

        if dim_values != class_nums:
            raise ValueError("dim_values must be equal to class_nums")

        weights = torch.ones(topk, dtype=torch.float32)
        # weights = nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32))
        self.register_buffer("weights", weights)

        self.dropout_layer = nn.Dropout1d(p=float(self.ff_dropout))
        # self.linear = nn.Sequential(nn.Linear(class_nums, 128),
        #                             nn.ReLU(),
        #                             nn.Linear(128, class_nums))

    def forward(self, x):
        x = einops.rearrange(x, "b c t v ... -> b c t (...) v")
        x = einops.rearrange(x, "b c t ... v -> b (... c) t v")
        x = torch.einsum("b n t v, t -> b n v", x, self.weights)
        x = self.dropout_layer(x)
        x = einops.reduce(x, "b n v -> b v", "mean")
        return x


def get_key_value_pairs_per_codebook(scaling_mode, num_channels, num_pairs, num_books):
    if scaling_mode == "constant_num_keys":
        num_channels, h, w = num_channels, 1, 1
        key_value_pairs_per_codebook = round(
            num_pairs * num_channels / num_books
        )
    elif scaling_mode == "free_num_keys":
        key_value_pairs_per_codebook = num_pairs
    else:
        raise NotImplementedError(f"Not implemented mode")
    return key_value_pairs_per_codebook

def get_threshold_ema_dead_code(num_channels, t_mode, scaling_mode, num_pairs, 
                                num_books, threshold_factor, batch_size):
    num_channels, h, w = num_channels, 1, 1 # output shapes
    if t_mode == "uniform_importance":
        num_pairs = get_key_value_pairs_per_codebook(scaling_mode, num_channels, num_pairs, num_books)
        threshold = threshold_factor * batch_size * h * w / num_pairs
    else:
        raise NotImplementedError(f"args.t_mode = {threshold}")
    return threshold


def get_decoder_module(num_codebooks, dim_values, dim_keys, num_channels, dataset_name, add_distance_to_values,
                       decoder_model, method):
    class_nums = get_class_nums(dataset_name)
    decoder_modules = []

    if add_distance_to_values:
        dim_values = dim_values + 1  # append distance to retrieved value

    if decoder_model == "linear-probe":
        dim_value_in = num_codebooks * dim_values

        if "mlp" in method:
            dim_value_in = num_channels

        decoder_modules += [
            Rearrange("b ... -> b (...)"),
            nn.Linear(dim_value_in, class_nums),
        ]
    elif decoder_model == "mlp-128":
        dim_value_in = num_codebooks * dim_values
        
        if "mlp" in method:
            dim_value_in = num_channels
            
        decoder_modules += [
            Rearrange("b ... -> b (...)"),
            nn.Linear(dim_value_in, 128),
            nn.ReLU(),
            nn.Linear(128, class_nums),
        ]
    elif decoder_model == "lp-no-bias":
        dim_value_in = num_codebooks * dim_values

        if "mlp" in method:
            dim_value_in = num_channels

        decoder_modules += [
            Rearrange("b ... -> b (...)"),
            nn.Linear(dim_value_in, class_nums, bias=False),
        ]
    elif decoder_model == "codebook-voting-logits":
        decoder = CodebookVotingLogitsDecoder(dim_values, class_nums, topk=3, ff_dropout=0.0)
        return decoder
    else:
        raise NotImplementedError(
            "Decoder size {} not supported".format(decoder_model)
        )
    decoder = nn.Sequential(*decoder_modules)
    return decoder


class ModelWrapper(nn.Module):
    def __init__(self, bottlenecked_encoder, decoder, method, ff_dropout, decoder_model, add_distance_to_values=False):
        super(ModelWrapper, self).__init__()
        self.bottlenecked_encoder = bottlenecked_encoder
        self.bottlenecked_encoder.freeze_encoder()
        self.decoder = decoder
        self.method = method
        self.ff_dropout = ff_dropout
        self.decoder_model = decoder_model
        self.add_distance_to_values = add_distance_to_values
        # self.values_interact_x = values_interact_x(70)
        if self.method == "ours":
            self.tuple_pos = 0  # this is the position of the returned value codes
        elif self.method == "kv_tune_full_decoder":
            self.tuple_pos = 0  # this is the position of the returned value codes
        elif self.method == "mlp":
            self.tuple_pos = -1  # this is the position of the returned key codes
            # self.tuple_pos = 0  # this is the position of the returned value codes
        else:
            raise NotImplementedError("Method {} not supported".format(method))
        self.dropout_layer = nn.Dropout1d(p=float(self.ff_dropout))

    def forward(self, x, adj, x_idx, cl_train):
        if self.method == "mlp":
            if not isinstance(self.bottlenecked_encoder.encoder, nn.Identity):
                bottleneck_tuple = self.bottlenecked_encoder(x, adj, x_idx, cl_train)[self.tuple_pos]
                # bottleneck_tuple = einops.rearrange(bottleneck_tuple, 'n m k b -> n (m k b)')
                # x = bottleneck_tuple.clone().detach() # 梯度停传
                x = bottleneck_tuple.clone()
            x = self.decoder(x)
        else:
            bottleneck_tuple = self.bottlenecked_encoder(x, adj, x_idx, cl_train)
            x = bottleneck_tuple[self.tuple_pos]
            if self.decoder_model == "codebook-voting-logits":
                x = self.decoder(x)
                return x, bottleneck_tuple[-1], bottleneck_tuple[-2], bottleneck_tuple[1], bottleneck_tuple[-3]
            if len(x.shape) == 4:
                x = einops.rearrange(x, "b c t v -> b (c t) v")
            if len(x.shape) == 5:
                x = einops.rearrange(x, "b c v h w -> b (c h w) v")
            if len(x.shape) == 6:
                x = einops.rearrange(x, "b c t v h w -> b (c t h w) v")
            # x = self.decoder(x)
            x = self.dropout_layer(x)
            if self.add_distance_to_values:
                distances = bottleneck_tuple[3]
                x = torch.cat((x, distances), dim=2)
        return x

    def forward_decoder(self, x):
        if self.method == "mlp":
            x = self.decoder(x)
        else:
            x = self.dropout_layer(x)
            x = self.decoder(x)
        return x

    def freeze_for_adaptation(self):
        if self.method == "ours":
            for param in self.decoder.parameters():
                param.requires_grad = False
        elif self.method == "kv_tune_full_decoder":
            return
        elif self.method == "mlp":
            return
        else:
            raise NotImplementedError(
                "Method {} not supported".format(self.method)
            )


################# test model wrapper #################  
class test_model_wrapper(nn.Module):
    def __init__(self, encoder, projection, decoder, method, ff_dropout):
        super(test_model_wrapper, self).__init__()
        self.encoder = encoder
        self.projection = projection
        self.decoder = decoder
        self.ff_dropout = ff_dropout
        self.method = method
        if self.method == "ours":
            self.tuple_pos = 0  # this is the position of the returned value codes
        elif self.method == "kv_tune_full_decoder":
            self.tuple_pos = 0  # this is the position of the returned value codes
        elif self.method == "mlp":
            self.tuple_pos = -1  # this is the position of the returned key codes
        else:
            raise NotImplementedError("Method {} not supported".format(method))
        self.dropout_layer = nn.Dropout1d(p=float(self.ff_dropout))

    def forward(self, x, adj):
        x = self.encoder(x, adj)
        x = self.projection(x)
        x = self.decoder(x)
        return x
    
class mlp(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim):
        super(mlp, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = einops.rearrange(x, "b c v -> b (c v)")
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x