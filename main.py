import gym
import d4rl
import numpy as np
import datetime
import torch
from tqdm import tqdm
import pickle
import os

from submodule.dynamics_ensemble import DynamicsEnsemble
from submodule.dynamics import PseudoDynamics
from submodule.iteration import PolicyIteration
from submodule.replay_memory import ReplayMemory
from submodule.sequential_replay import SeqReplayMemory
from utilities.currency_exchange import CurrencyExchange
from utilities.arguments import args_setting, print_args
from utilities.utils import setup_seed, Evaluator, get_free_gpu
import utilities.static
import os

args = args_setting()

if args.task == 'CurrencyExchange':
    env = CurrencyExchange()
    folder_path = 'download/currencyexchange-random-v0.pkl'
    with open(folder_path, 'rb') as f:
        dataset = pickle.load(f)
        done_func = utilities.static['dummy'].termination_fn
        reward_func = None
elif "antmaze" in args.task:
    env = gym.make(args.task, reward_type='sparse')
    dataset = d4rl.qlearning_dataset(env.unwrapped)
    reward_func = None
    done_func = utilities.static['dummy'].termination_fn
else:
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env.unwrapped)
    reward_func = None
    done_func = utilities.static[args.task.split('-')[0]].termination_fn

env.seed(args.seed)
setup_seed(args.seed)

state = dataset['observations']
action = dataset['actions']
next_state = dataset['next_observations']
reward = np.expand_dims(np.squeeze(dataset['rewards']), 1)
done = np.expand_dims(np.squeeze(dataset['terminals']), 1)
state_dim = env.observation_space.shape[0]
action_space = env.action_space
action_dim = action_space.shape[0]

if args.cpu:
    print("Use CPU for Training")
    args.device = torch.device('cpu')
else:
    free_gpu = get_free_gpu() 
    print(f"Index of GPU used currently:{free_gpu}")
    args.device = torch.device(f"cuda:{free_gpu}")

print_args(args)
print(f"\n--------------------------------------------------")
print(f"Task: {args.task}")
print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
print(f"Dataset size: {len(state)}")
print(f"--------------------------------------------------\n")

predict_reward = reward_func is None

ensemble_dynamics = DynamicsEnsemble(state_dim, action_space.shape[0], predict_reward, args)
ensemble_dynamics.train({'obs':state, 'act':action, 'obs_next':next_state, 'rew':reward})
transition = ensemble_dynamics.transition

real_seq_memory = SeqReplayMemory(state.shape[0])
real_memory = ReplayMemory(state.shape[0])
normalized_reward = transition.get_normalized_reward(reward)
real_memory.push(state, action, normalized_reward, next_state, done)
real_seq_memory.push(state, action, normalized_reward, next_state, done)
offline_agent = PolicyIteration(state_dim, action_space, real_memory, real_seq_memory, args, value_clip=predict_reward)

all_state = np.concatenate([state, next_state[~done.astype(bool).reshape([-1])]], axis=0)
adv_dyna = PseudoDynamics(all_state, offline_agent, transition, args, done_func, reward_func)
test_adv_dyna = PseudoDynamics(all_state, offline_agent, transition, args, done_func, reward_func)
dynamics_log = ("k_" + str(args.order_transition) if not args.dynamics_reweight else ("reweight_" + str(args.num_sample_transition) + '_' + str(args.topk))) + args.tuning_info
data_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

evaluator = Evaluator(env, test_adv_dyna, offline_agent, [0, 1, 2, 3, 4], data_time, args)
num_thounsands = 0
num_updates = 0
critic_loss = 0
policy_loss = 0
state = adv_dyna.reset()

while 1:
    total_train_time = 0
    total_eval_time = 0
    if args.eval is True:
        evaluator.eval(num_updates, critic_loss, policy_loss)
    for i_rollout in range(1000):
        action = offline_agent.act(state)
        _, adv_q, _, _ = adv_dyna.step(action)
        critic_loss, policy_loss = offline_agent.offline_update_parameters(state, action, adv_q)
        num_updates += 1
        state = adv_dyna.state.cpu().numpy()

    num_thounsands += 1
    if num_updates >= args.agent_num_steps:
        break

env.close()
