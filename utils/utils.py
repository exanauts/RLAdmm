import torch
import numpy as np


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def index_from_list(listA, indexA):
    res = []
    for i in indexA:
        res.append(listA[i])
    return np.array(res)

def huber_loss(X,Y):
    err = X-Y
    loss = torch.where(torch.abs(err) < 1.0, 0.5 * torch.pow(err,2), torch.abs(err) - 0.5).mean()
    return loss

def get_closestmatch_index(listA, elementA):
    return np.argmin(np.absolute(np.array(listA)-elementA))


def mismatch_loss(action1, action2):
    # action1 and action2 are assumed to be arrays
    return (a!=b).float()