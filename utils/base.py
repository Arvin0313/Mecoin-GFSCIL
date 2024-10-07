import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_class_nums(dataset_name):
    if dataset_name == "CoraFull":
        class_nums = 70
    elif dataset_name == "Reddit":
        class_nums = 41
    elif dataset_name == "CS":
        class_nums = 15
    elif dataset_name == "Computers":
        class_nums = 10
    else:
        raise NotImplementedError("Dataset {} not supported".format(dataset_name))
    return class_nums


class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, outputs_student, outputs_teacher):
        # Apply softmax to teacher and student logits
        p_teacher = F.softmax(outputs_teacher / self.temperature, dim=1)
        p_student = F.softmax(outputs_student / self.temperature, dim=1)
        
        # Calculate the cross-entropy loss between student and teacher predictions
        loss = F.kl_div(p_student.log(), p_teacher, reduction='batchmean') * (self.temperature**2)
        
        return loss
