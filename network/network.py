import torch
import torch.nn as nn
import numpy as np
import pdb

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU())



        self.fc1_critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU())
        self.fc2_critic = nn.Linear(128, 1)


        self.fc1_actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU())
        self.fc2_actor = nn.Linear(128, 2)

        self.common_layers = [self.fc1, self.fc2]
        self.fc_critic_layers = [self.fc1_critic, self.fc2_critic]
        self.fc_actor_layers = [self.fc1_actor, self.fc2_actor]
        # self.device = device

    def forward(self, x):
        # assume input: tensors (NCHW); output: tensors

        batch_size = x.shape[0]
        for layer in self.common_layers:
            x = layer(x)

        x_actor = x
        x_critic = x

        for layer in self.fc_actor_layers:
            x_actor = layer(x_actor)

        for layer in self.fc_critic_layers:
            x_critic = layer(x_critic)

        return x_critic, x_actor





class Actor(nn.Module):
    def __init__(self, dim_state, dim_action, activation, coef_add, coef_multiply, hidden=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim_state, hidden),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU())

        self.fc1_actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU())

        self.coef_add = coef_add
        self.coef_multiply = coef_multiply

        if activation=='Sigmoid':
            self.fc2_actor = nn.Sequential(
                nn.Linear(hidden, dim_action),
                nn.Sigmoid())
        elif activation=='ReLU':
            self.fc2_actor = nn.Sequential(
                nn.Linear(hidden, dim_action),
                nn.ReLU())
        elif activation=='None':
            self.fc2_actor = nn.Linear(hidden, dim_action)
        else:
            raise NotImplementedError

        self.layers = [self.fc1, self.fc2, self.fc1_actor, self.fc2_actor]

    def forward(self, x):
        batch_size = x.shape[0]
        for layer in self.layers:
            x = layer(x)
            
        return x * self.coef_multiply + self.coef_add

    def saturation_loss(self, x):
        batch_size = x.shape[0]
        for layer in self.layers:
            x = layer(x)
            
        return (torch.tan(x*np.pi/2.)+torch.tan((1-x)*np.pi/2.)).mean()


class Critic(nn.Module):
    def __init__(self, dim_state, dim_action, hidden=64, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(dim_state, hidden)
        self.fc2 = nn.Linear(hidden+dim_action, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.init_weights(init_w)
        
    # def __init__(self, dim_state, dim_action, hidden=64, init_w=3e-3):
    #     super(Critic, self).__init__()
    #     self.fc1 = nn.Linear(dim_state, hidden, bias=False)
    #     self.fc2 = nn.Linear(hidden+dim_action, hidden, bias=False)
    #     self.fc3 = nn.Linear(hidden, 1, bias=False)
    #     self.relu = nn.ReLU()
    #     self.dim_state = dim_state
    #     self.dim_action = dim_action
    #     self.init_weights(init_w)
          


    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs[:,:self.dim_state], xs[:,self.dim_state:]

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out






class DQN(nn.Module):
    def __init__(self, dim_state, dim_action, hidden=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim_state, hidden),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU())


        self.fc3 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU())

        self.fc4 = nn.Linear(hidden, dim_action)

        self.layers_common = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x):
        # batch_size = x.shape[0]
        for layer in self.layers_common:
            x = layer(x)

        return x