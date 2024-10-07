import torch


def get_optimizer(model, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError("Optimizer {} not supported".format(args.optimizer))
    return optimizer
