import math
import torch
from torch.functional import F
import numpy as np
import random
from prettytable import PrettyTable
import subprocess
import os
import d4rl
from utilities.arguments import print_args

def get_free_gpu():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
    )
    gpu_memory = [int(x) for x in result.decode("utf-8").strip().split("\n")]
    return gpu_memory.index(min(gpu_memory))  # Return the index of the GPU with the least memory usage

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def soft_clamp(x : torch.Tensor, _min=None, _max=None, _floor_std=None):

    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    if _floor_std is not None:
        x = torch.log(torch.exp(x) + _floor_std)
    return x


def create_log_gaussian(mean, log_std, t):

    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)

    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def log_dynamic_loss(args, iteration, prob_loss, mse_loss):
    if not os.path.exists(args.log_loss_path):
            os.makedirs(args.log_loss_path)
    with open(args.log_loss_path + args.dynamics_model_name + '_' + args.task + '_' + str(args.transition_layer_size) + '_batch' + str(args.transition_batch_size) +'.txt', 'a') as file:
        file.write(f"Epoch:{iteration + 1}|ProbailityLoss:{prob_loss}|MSELoss:{mse_loss}\n")

def log_belief_surprise(args, belief, surprise):
    dynamics_log = "belief_" + str(args.num_sample_transition) + args.tuning_info
    bregman_log = "_bregman_" if args.bregman_reg else ""
    info_path = args.log_policy_info + args.dynamics_model_name + '_' + args.task + '_' + str(args.transition_layer_size) + '_batch' + \
                    str(args.transition_batch_size) + '_agent' + bregman_log + str(args.agent_layer_size) + '_' + str(args.adv_batch_size) \
                    + '_' + str(args.real_batch_size) + '_' + str(args.seed) + '_' + dynamics_log 
    np.save(info_path + '_belief.npy', belief)
    np.save(info_path + '_surprise.npy', surprise)


class Evaluator(object):
    def __init__(self, real_env, adv_env, agent, mode, data_time, args):
        self.real_env = real_env
        self.adv_env = adv_env
        self.agent = agent
        self.mode = mode
        self.optim = args.optim
        self.steps = args.eval_steps # default 1k
        self.adv_steps = self.steps if self.steps <= args.adv_horizon else args.adv_horizon
        self.record_time = args.record_time
        self.task_name = args.task
        self.is_stochastic = 'HIV' in self.task_name or 'Currency' in self.task_name
        if self.is_stochastic:
            self.cvar_alpha = 0.1
            print(f"Stochastic Environment: ('{self.task_name}').")
        else:
            print(f"D4RL Environment: ('{self.task_name}').")
        self.bregman_log = "_bregman_" if args.bregman_reg else ""
        if args.use_adaptive_belief:
            ablation_info =  "_ablation" if args.ablation_belief else ""
            self.dynamics_log = "belief_" + str(args.num_sample_transition) + ablation_info + '_' + self.optim + args.tuning_info
        else:
            self.dynamics_log = ("k_" + str(args.order_transition) if not args.dynamics_reweight 
                                 else ("reweight_" + str(args.num_sample_transition) + '_' + str(args.topk))) \
                                 + '_' + self.optim + args.tuning_info
        self.path = args.log_policy_info + args.dynamics_model_name + '_' + args.task + '_' + str(args.transition_layer_size) + '_batch' + \
                    str(args.transition_batch_size) + '_agent' + self.bregman_log + str(args.agent_layer_size) + '_' + str(args.adv_batch_size) \
                    + '_' + str(args.real_batch_size) + '_' + str(args.seed) + '_' + self.dynamics_log + '.txt'
        self.info_path = args.log_policy_info + args.dynamics_model_name + '_' + args.task + '_' + str(args.transition_layer_size) + '_batch' + \
                    str(args.transition_batch_size) + '_agent' + self.bregman_log + str(args.agent_layer_size) + '_' + str(args.adv_batch_size) \
                    + '_' + str(args.real_batch_size) + '_' + str(args.seed) + '_' + self.dynamics_log + '_info.txt'
        print_args(args, self.path)

    def eval(self, num_updates, critic_loss, policy_loss):
        table = PrettyTable()
        if self.is_stochastic:
            table.add_column('Metric', ['Real Env Mean', f'Real Env CVaR@{self.cvar_alpha}', 'Adv Env Mean'])
        else:
            table.add_column('Policy Type', ['Real Env', 'Adv Env', 'Normalized Score'])

        for temp_mode in self.mode:
            reward_reg = []
            total_step = 0
            while total_step < self.steps:
                i_step = 0
                state = self.real_env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = self.agent.act(np.float32(state), mode=temp_mode)
                    state, reward, done, _ = self.real_env.step(action.ravel())
                    episode_reward += reward
                    i_step += 1
                total_step += i_step
                reward_reg.append(episode_reward)
            if "antmaze" in self.task_name:
                final_x = state[0]
                final_y = state[1]
                print(f"Final (x, y) Coords: ({final_x:.4f}, {final_y:.4f}). Target Goal: ({self.real_env.target_goal})")
            
            num_episode = 200 if self.is_stochastic else 10
            state = np.concatenate([self.real_env.reset().reshape([1, -1]) for _ in range(num_episode)], 0)
            self.adv_env.reset(np.float32(state))
            index = np.arange(num_episode)
            episode_reward = np.zeros(num_episode)
            adv_q_target_reg = []
            adv_q_reg = []
            ensemble_std_reg = []
            total_inference_time = 0
            
            for _ in range(self.adv_steps):
                if self.record_time:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record() 
                    action = self.agent.act(self.adv_env.state, mode=temp_mode)
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    total_inference_time += elapsed_time_ms
                else:
                    action = self.agent.act(self.adv_env.state, mode=temp_mode)
                feedback, adv_q_target, adv_q, ensemble_std = self.adv_env.step(action, test_mode=True)
                reward, _, done = feedback
                episode_reward[index] += self.adv_env.transition.get_raw_reward(reward)
                index = index[~done]
                adv_q_target_reg.append(adv_q_target.reshape(-1))
                adv_q_reg.append(adv_q.reshape(-1))
                ensemble_std_reg.append(ensemble_std)
                if index.shape[0] == 0:
                    break

            real_reward = np.mean(reward_reg)
            adv_reward = episode_reward.mean()
            if self.is_stochastic:
                rewards_sorted = np.sort(reward_reg)
                cvar_idx = int(len(reward_reg) * self.cvar_alpha)
                cvar_reward = np.mean(rewards_sorted[:cvar_idx]) if cvar_idx > 0 else real_reward
                column_data = [round(real_reward, 4), round(cvar_reward, 4), round(adv_reward, 4)]
                table.add_column(str(temp_mode), column_data)
                adv_q_target_reg = np.concatenate(adv_q_target_reg, 0)
                adv_q_reg = np.concatenate(adv_q_reg, 0)
                ensemble_std_reg = np.concatenate(ensemble_std_reg, 0)
                std_adv_q_target, mean_adv_q_target = np.std(adv_q_target_reg), np.mean(adv_q_target_reg)
                std_adv_q, mean_adv_q = np.std(adv_q_reg), np.mean(adv_q_reg)
                ensemble_logstd_reg = np.log(ensemble_std_reg)
                std_ensemble_logstd, mean_ensemble_logstd = np.std(ensemble_logstd_reg), np.mean(ensemble_logstd_reg)

                mean_adv_q_target = mean_adv_q_target
                std_adv_q_target = std_adv_q_target

                mean_adv_q = mean_adv_q
                std_adv_q = std_adv_q

                mean_ensemble_logstd = mean_ensemble_logstd
                std_ensemble_logstd = std_ensemble_logstd
                string_info = "Update number:" + str(num_updates) +\
                        "\nstd_ensemble_logstd:" + str(std_ensemble_logstd) + "|mean_ensemble_logstd:"+str(mean_ensemble_logstd) +\
                        "|mean_adv_q:" + str(mean_adv_q) + "|std_adv_q:" + str(std_adv_q) +\
                        "|mean_adv_q_target:" + str(mean_adv_q_target) + "|std_adv_q_target:" + str(std_adv_q_target) +\
                        "|critic_loss:" + str(critic_loss) + "|policy_loss:" + str(policy_loss)
            else:
                score = self.real_env.get_normalized_score(real_reward)
                if temp_mode == 1:
                    table.add_column('1 (Reported)', [round(real_reward, 4), round(adv_reward, 4), round(score, 4)])

                    adv_q_target_reg = np.concatenate(adv_q_target_reg, 0)
                    adv_q_reg = np.concatenate(adv_q_reg, 0)
                    ensemble_std_reg = np.concatenate(ensemble_std_reg, 0)
                    std_adv_q_target, mean_adv_q_target = np.std(adv_q_target_reg), np.mean(adv_q_target_reg)
                    std_adv_q, mean_adv_q = np.std(adv_q_reg), np.mean(adv_q_reg)
                    ensemble_logstd_reg = np.log(ensemble_std_reg)
                    std_ensemble_logstd, mean_ensemble_logstd = np.std(ensemble_logstd_reg), np.mean(ensemble_logstd_reg)

                    mean_adv_q_target = mean_adv_q_target
                    std_adv_q_target = std_adv_q_target

                    mean_adv_q = mean_adv_q
                    std_adv_q = std_adv_q

                    mean_ensemble_logstd = mean_ensemble_logstd
                    std_ensemble_logstd = std_ensemble_logstd
                    if self.record_time:
                        inference_time = total_inference_time / self.steps
                        string_info = "Update number:" + str(num_updates) +\
                            "\nstd_ensemble_logstd:" + str(std_ensemble_logstd) + "|mean_ensemble_logstd:"+str(mean_ensemble_logstd) +\
                            "|mean_adv_q:" + str(mean_adv_q) + "|std_adv_q:" + str(std_adv_q) +\
                            "|mean_adv_q_target:" + str(mean_adv_q_target) + "|std_adv_q_target:" + str(std_adv_q_target) +\
                            "|critic_loss:" + str(critic_loss) + "|policy_loss:" + str(policy_loss) +\
                            "|inference time:" + str(inference_time)
                    else:
                        string_info = "Update number:" + str(num_updates) +\
                            "\nstd_ensemble_logstd:" + str(std_ensemble_logstd) + "|mean_ensemble_logstd:"+str(mean_ensemble_logstd) +\
                            "|mean_adv_q:" + str(mean_adv_q) + "|std_adv_q:" + str(std_adv_q) +\
                            "|mean_adv_q_target:" + str(mean_adv_q_target) + "|std_adv_q_target:" + str(std_adv_q_target) +\
                            "|critic_loss:" + str(critic_loss) + "|policy_loss:" + str(policy_loss)
                else:
                    table.add_column(str(temp_mode), [round(real_reward, 4), round(adv_reward, 4), round(score, 4)])
        with open(self.path, 'a') as f:
            print(f"Update number:{num_updates}", file=f)
            print(table, file=f)
        with open(self.info_path, 'a') as f:
            print(string_info, file=f)