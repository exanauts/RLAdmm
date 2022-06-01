import os
from network.agent_DQN import DeepAgent
import argparse
import json
import pdb
import julia
import time
from network.Memory import Memory
import numpy as np
import datetime
import copy
import matplotlib.pyplot as plt
from utils.utils import *



if __name__ == '__main__':


    ############################ initialize agents and start training ############################
    

    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--num_case', type=int, default=9, help='case 9 or case 30')
    parser.add_argument('--gamma', type=float, default=.99, help='discount factor')
    parser.add_argument('--max_episodes', type=int, default=500, help='number of training episodes')
    parser.add_argument('--buffer_len', type=int, default=50000, help='length of replay buffer')
    parser.add_argument('--wait_before_train', type=int, default=20, help='the number of episodes to wait before training starts')
    parser.add_argument('--max_iters', type=int, default=500, help='number of iterations per episode')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--change_rho_interval', type=int, default=50, help='adjust the rho value every change_rho_interval ADMM iterations')
    parser.add_argument('--reward_interval', type=int, default=20, help='primal and dual residual are averaged in a window of length reward_interval')
    parser.add_argument('--state_interval', type=int, default=20, help='state contains the history of length state_interval')
    parser.add_argument('--update_target_interval', default=10, type=int, help='episode interval of assigning Q network to Q target network')
    parser.add_argument('--multiplier', default=1., type=float, help='exploration parameter')
    parser.add_argument('--activation_actor', type=str, default='Sigmoid', help='activation function of the last layer of the actor ["Sigmoid","ReLU","None"]')
    parser.add_argument('--coef_add_actor', type=float, default=100., help='additive constant in the last layer of the actor')
    parser.add_argument('--coef_multiply_actor', type=float, default=1000., help='multiplicative constant in the last layer of the actor')
    parser.add_argument('--weight_expert_0', type=float, default=2e-5, help='weight of the loss used to encourage the output of the policy to be close to the expert policy')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    # parser.add_argument('--lr_decay', type=str, default='None', help='learning rate decay [None, inv (1/k), inv_root (1/sqrt(k))]')
    parser.add_argument('--lr_adjust_interval', type=int, default=100, help='number of episodes between learning rate decays')
    parser.add_argument('--loss_clip_disable', action='store_true', help='if not clip loss, raise this flag')
    parser.add_argument('--loss_clip_mag', type=float, default=.1, help='max norm of clipped gradients')
    parser.add_argument('--save_interval', type=int, default=10, help='number of episodes between successive model weights save')
    parser.add_argument('--use_baseline', action='store_true', help='whether to compare with baseline model in reward computation')

    args = parser.parse_args()

    discrete_actions_pq = [100,200,300,400,500,600,700,800,900,1000]
    discrete_actions_va = [500, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]

    dim_action = len(discrete_actions_pq)
    if args.num_case==9:
        ngen = 3 # 9 bus
        nline = 9 # 9 bus
    elif args.num_case==30:
        ngen = 6 # 30 bus
        nline = 41 # 30 bus
    # elif args.num_case==118:
    #     ngen = 54 # 118 bus
    #     nline = 186 # 118 bus
    else:
        raise NotImplementedError
    dim_rho_pq = 2*ngen+4*nline
    dim_rho_va = 2*nline
    dim_rho = dim_rho_pq+dim_rho_va
    args = parser.parse_args()
    use_baseline = args.use_baseline


    lr = args.lr
    loss_clip = not args.loss_clip_disable

    data_path = "/data/case9"
    dir_path = ('trained_models/DQN_entrywise/case{}/{}/'.format(args.num_case,datetime.datetime.now())).replace(':','_').replace(' ','_')

    network_path_pq = os.path.join(dir_path, 'agent_pq')
    network_path_va = os.path.join(dir_path, 'agent_va')
    if not os.path.exists(network_path_pq):
        os.makedirs(network_path_pq)    
    if not os.path.exists(network_path_va):
        os.makedirs(network_path_va)
    
    # save input arguments into json file
    with open(os.path.join(dir_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    log_path = os.path.join(dir_path, 'log.txt')
    f = open(log_path, 'w')



    # initialize replay memory and RL agent
    ReplayMemory_pq = Memory(args.buffer_len)
    ReplayMemory_va = Memory(args.buffer_len)
    agent_pq = DeepAgent(network_path_pq, lr, args, dim_action)
    agent_va = DeepAgent(network_path_va, lr, args, dim_action)



    # use Julia ADMM solver
    try:
        j = julia.Julia()
    except:
        from julia.api import Julia
        Julia(compiled_modules=False)
        j = julia.Julia()
    environment = j.include('RL_environment_V2.jl')




    #############################################################
    #---------------------Training Deep RL----------------------#
    #############################################################

    # Initialize 
    running_reward = 0
    episode_count = 0
    history_cum_reward = []
    history_actions = []
    history_converge_iters = []


    train_time_start = time.time()

    # Run training until break
    multiplier = args.multiplier
    while episode_count < args.max_episodes:


        converge_counter = 0
        prim_thres_counter = 0
        dual_thres_counter = 0

        
        # Intialize the iteration arrays
        history_iter_actions = []
        history_primres = []
        history_dualres = []


        # Initialize the ADMM inputs
        param = None
        info = None
        tau = None
        

        converge_iters = args.max_iters # number of ADMM iterations until convergence
        prev_rho_reward_list = []

        if episode_count>0 and episode_count%args.lr_adjust_interval==0:
            lr = lr/2.
            agent_pq.reset_lr(lr)
            agent_va.reset_lr(lr)

        if episode_count % args.update_target_interval==0:
            agent_pq.assign_target()
            agent_va.assign_target()

        cum_reward = 0
        for t in range(args.max_iters+1):
            iter_time_start = time.time()

            # Select action
            if t==0:
                action_idx_pq = np.array([3]*dim_rho_pq)
                action_idx_va = np.array([7]*dim_rho_va)
            elif t>=2*args.state_interval:
                state = np.vstack([history_primres[-args.state_interval:],history_dualres[-args.state_interval:]]).T
                state_pq = state[:dim_rho_pq,:]
                state_va = state[dim_rho_pq:,:]
                if t%args.change_rho_interval == 0:
                    action_idx_pq = agent_pq.action_selection(state_pq)
                    action_idx_va = agent_va.action_selection(state_va)
                    eps_greedy = np.maximum(1-0.003*((episode_count+1)**1.),0.02)
                    for ii in range(dim_rho_pq):
                        if np.random.rand()<eps_greedy:
                            action_idx_pq[ii] = np.random.choice(range(dim_action))
                    for ii in range(dim_rho_va):
                        if np.random.rand()<eps_greedy:
                            action_idx_va[ii] = np.random.choice(range(dim_action))


            #history_iter_actions.append(index_from_list(discrete_actions, action_idx))
            
            time_start_env = time.time()
            # Apply action to environment and receive reward
            if use_baseline and t>=2*args.state_interval:
                # new_res_baseline, converge_baseline, prim_thres_baseline, dual_thres_baseline, param_baseline, info_baseline, tau_baseline = environment(t, 500., 500., param, info, tau)
                new_res_baseline, converge_baseline, prim_thres_baseline, dual_thres_baseline, param_baseline, info_baseline, tau_baseline = environment(t, 400., 4000., param, info, tau)
                # new_res_baseline, converge_baseline, prim_thres_baseline, dual_thres_baseline, param_baseline, info_baseline, tau_baseline = environment(t, action_pq_prev, action_va_prev, param, info, tau)
                primal_res_baseline, dual_res_baseline = param_baseline.rp, param_baseline.rd
            
            action_pq, action_va = index_from_list(discrete_actions_pq, action_idx_pq), index_from_list(discrete_actions_va, action_idx_va)
            new_res, converge, prim_thres, dual_thres, param, info, tau = environment(t, action_pq, action_va, param, info, tau)
            primal_res, dual_res = param.rp, param.rd
            history_primres.append(copy.deepcopy(param.rp))
            history_dualres.append(copy.deepcopy(param.rd))
            time_end_env = time.time()
            print('Time for environment transition: {}'.format(time_end_env-time_start_env))
            # print('policy:', primal_res, dual_res)
            


            if t>=2*args.state_interval:
                # iter_reward = (np.absolute(primal_res_baseline)-np.absolute(primal_res))/(np.absolute(primal_res_baseline)+1e-6)+(np.absolute(dual_res_baseline)-np.absolute(dual_res))/(np.absolute(dual_res_baseline)+1e-6)
                if use_baseline:
                    iter_reward1 = (dim_rho*primal_res**2>info.eps_pri**2)*(np.absolute(primal_res_baseline)-np.absolute(primal_res))/(np.absolute(primal_res_baseline)+1e-6)
                    iter_reward2 = (dim_rho*dual_res**2>info.eps_dual**2)*(np.absolute(dual_res_baseline)-np.absolute(dual_res))/(np.absolute(dual_res_baseline)+1e-6)
                else:
                    iter_reward1 = (dim_rho*primal_res**2>info.eps_pri**2)*(np.absolute(history_primres[-2])-np.absolute(primal_res))/(np.absolute(history_primres[-2])+1e-6)
                    iter_reward2 = (dim_rho*dual_res**2>info.eps_dual**2)*(np.absolute(history_dualres[-2])-np.absolute(dual_res))/(np.absolute(history_dualres[-2])+1e-6)


                iter_reward = iter_reward1 + iter_reward2

                iter_reward = np.clip(iter_reward, a_min=-1., a_max=1.)
                iter_reward += 200. if converge else 0.

                cum_reward = args.gamma * cum_reward + np.mean(iter_reward)

                # # encourage smooth change of rho
                # iter_reward += 0.1*(-(action_pq-history_iter_actions_pq[-2])**2-(action_va-history_iter_actions_va[-2])**2)



                # state = np.vstack([history_primres[-args.state_interval-1:-1],history_dualres[-args.state_interval-1:-1]]).T
                new_state = np.vstack([history_primres[-args.state_interval:],history_dualres[-args.state_interval:]]).T
                new_state_pq = new_state[:dim_rho_pq,:]
                new_state_va = new_state[dim_rho_pq:,:]

                if converge or t==args.max_iters:
                    terminal = 1
                else:
                    terminal = 0
                
                time_start_RL = time.time()

                data_tuple_list = []
                for ii in range(dim_rho_pq):
                    data_tuple = (state_pq[ii,:], action_idx_pq[ii], new_state_pq[ii,:], iter_reward[ii], terminal)
                    data_tuple_list.append(data_tuple)
                errs = np.absolute(agent_pq.compute_TD(data_tuple_list).cpu().numpy())
                for ii in range(dim_rho_pq):
                    ReplayMemory_pq.add(errs[ii], data_tuple_list[ii])

                for ii in range(dim_rho_va):
                    data_tuple = (state_va[ii,:], action_idx_va[ii], new_state_va[ii,:], iter_reward[ii+dim_rho_pq], terminal)
                    data_tuple_list.append(data_tuple)
                errs = np.absolute(agent_va.compute_TD(data_tuple_list).cpu().numpy())
                for ii in range(dim_rho_pq):
                    ReplayMemory_va.add(errs[ii], data_tuple_list[ii])


                # Train RL agent
                if episode_count>=args.wait_before_train:
                    idxs, TDs = agent_pq.train_n(ReplayMemory_pq, args.batch_size, loss_clip, args.loss_clip_mag, None)
                    errs = np.absolute(TDs.cpu().numpy())
                    for i in range(args.batch_size):
                        ReplayMemory_pq.update(idxs[i], errs[i])

                    idxs, TDs = agent_va.train_n(ReplayMemory_va, args.batch_size, loss_clip, args.loss_clip_mag, None)
                    errs = np.absolute(TDs.cpu().numpy())
                    for i in range(args.batch_size):
                        ReplayMemory_va.update(idxs[i], errs[i])

                time_end_RL = time.time()
                print('Time for policy update: {}'.format(time_end_RL-time_start_RL))

                iter_time_end = time.time()


                s_log = 'Iter: {}/{}      lr: {}      t={:<1.3f}      Reward: {}      Primal/Dual Res: ({},{})      Terminal: {}'.format(
                        t, episode_count,
                        lr,
                        iter_time_end - iter_time_start,
                        np.mean(iter_reward),
                        np.linalg.norm(history_primres[-1]), np.linalg.norm(history_dualres[-1]),
                        terminal)

                print(s_log)
                f.write(s_log+'\n')

            # Check termination
            if converge == True:
                converge_iters = t
                converge_counter += 1
                prim_thres_counter += 1
                dual_thres_counter += 1

                print('#######################################')
                print('Converged in {} Iterations'.format(converge_iters))
                print('#######################################')

                break
            elif t==args.max_iters:
                prim_thres_counter += prim_thres
                dual_thres_counter += dual_thres



                
        # #---------------------Save Results From Mini-Batch----------------------#
        history_converge_iters.append(converge_iters)
        history_actions.append(history_iter_actions)
        history_cum_reward.append(cum_reward)
        


        # Add to episode count
        episode_count += 1


        if episode_count%10==0:
            agent_pq.save_network(network_path_pq, episode_count)
            agent_va.save_network(network_path_va, episode_count)
            plt.plot(history_converge_iters)
            plt.savefig(os.path.join(dir_path, 'train.png'))

   
    train_time_end = time.time()
    
