import os
import numpy as np
from network.network import *
from utils.utils import *
import pdb

def huber_loss(X,Y):
    err = X-Y
    loss = torch.where(torch.abs(err) < 1.0, 0.5 * torch.pow(err,2), torch.abs(err) - 0.5).mean()
    return loss



class DeepAgent():
    def __init__(self, network_path, lr, args, dim_action, vec_rho=False, dim_rho=1):
        
        self.network_path = network_path
        self.gamma = args.gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if vec_rho==False:
            self.dim_state = 2*args.state_interval
        else:
            self.dim_rho = dim_rho
            self.dim_state = 2*args.state_interval*dim_rho
        self.dim_action = dim_action

        self.model = DQN(self.dim_state, self.dim_action).to(self.device)
        self.model_target = DQN(self.dim_state, self.dim_action).to(self.device)


        # sync network parameters
        self.assign_target()

        self.reset_lr(lr)

        print()


    def assign_target(self):
        hard_update(self.model_target, self.model)

    def reset_lr(self, lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas = (.9, .999))


    def Q_val(self, xs):
        xs = torch.Tensor(xs).to(self.device)
        return self.model(xs).detach().cpu().numpy()


    def action_selection(self, state):
        # pdb.set_trace()
        qvals = self.Q_val(state)

        if qvals.shape[0]>1:
            # Evaluating batch
            action = np.argmax(qvals, axis=1)
        else:
            # Evaluating one sample
            action = np.zeros(1)
            action[0]=np.argmax(qvals)

        return action.astype(int)

    def compute_TD(self, data_tuple_minibatch):
        batch_size = len(data_tuple_minibatch)
        
        states = []
        actions = []
        new_states = []
        rewards = []
        terminals = []
        for i in range(batch_size):
            states.append(data_tuple_minibatch[i][0])
            actions.append(data_tuple_minibatch[i][1])
            new_states.append(data_tuple_minibatch[i][2])
            rewards.append(data_tuple_minibatch[i][3])
            terminals.append(data_tuple_minibatch[i][4])
        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        new_states = torch.Tensor(new_states).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        terminals = torch.Tensor(terminals).to(self.device)

        Qval_curr = self.model(states)
        Qval_new = self.model(new_states)
        Qval_new_target = self.model_target(new_states)

        term_ind = torch.where(terminals==1)[0]
        nonterm_ind = torch.where(terminals==0)[0]

        TD = torch.zeros(batch_size).to(self.device)

        try:
            TD[nonterm_ind] = rewards[nonterm_ind] + self.gamma * Qval_new_target[nonterm_ind, torch.argmax(Qval_new[nonterm_ind], axis=1)] - Qval_curr[nonterm_ind, actions[nonterm_ind].long()]
        except:
            # nonterm_ind is empty
            TD[nonterm_ind] = rewards[nonterm_ind] + self.gamma * Qval_new_target[nonterm_ind, 0] - Qval_curr[nonterm_ind, actions[nonterm_ind].long()]
            
        TD[term_ind] = rewards[term_ind] - Qval_curr[term_ind, actions[term_ind].long()]

        return TD.detach()

    def train_n(self, ReplayMemory, batch_size, loss_clip, loss_clip_mag, data_tuple, TD_clip=True):
        if batch_size==1 and data_tuple is not None:
            train_batch = [data_tuple]
            idx = None
        else:
            batch = ReplayMemory.sample(batch_size)
            train_batch = [b[1] for b in batch]
            idx = [b[0] for b in batch]
        states = []
        actions = []
        new_states = []
        rewards = []
        terminals = []
        for i in range(batch_size):

            states.append(train_batch[i][0])
            actions.append(train_batch[i][1])
            new_states.append(train_batch[i][2])
            rewards.append(train_batch[i][3])
            terminals.append(train_batch[i][4])

        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).long().to(self.device)
        new_states = torch.Tensor(new_states).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        terminals = torch.Tensor(terminals).to(self.device)
        # batch_size = len(states)

        TDs = self.compute_TD(train_batch)
        if TD_clip:
            TDs = torch.clamp(TDs, min=-5., max=5.)

        # model update
        self.model.train()
        self.model.zero_grad()

        qs = self.model(states)[range(batch_size),actions]
        q_targets = qs.detach()+TDs
            
        # TD_loss = ((qs-q_targets)**2).mean()
        TD_loss = huber_loss(qs, q_targets)

        TD_loss.backward()
        if loss_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), loss_clip_mag)
        self.optimizer.step()
        self.model.eval()


        return idx, TDs



    def save_network(self, save_path, episode):
        torch.save(self.model.state_dict(), os.path.join(save_path,'model'+str(episode)+'.pth'))


    def load_network(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()
        


