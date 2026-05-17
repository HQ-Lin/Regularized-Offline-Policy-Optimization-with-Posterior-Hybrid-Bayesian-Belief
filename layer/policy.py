import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
import numpy as np

from utilities.utils import soft_clamp

epsilon = 1e-6
NOISE_STD = 0.2

LOG_SIG_MIN = -4
LOG_SIG_MAX = 2


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, BatchLinear):
        for i in range(m.batch):
            torch.nn.init.xavier_uniform_(m.weight[:, :, i], gain=1)
        torch.nn.init.constant_(m.bias, 0)


class BatchLinear(nn.Module):
    def __init__(self, batch: int, in_features: int, out_features: int, first=False) -> None:
        super(BatchLinear, self).__init__()
        self.batch = batch
        self.first = first

        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(out_features, in_features, batch)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(out_features, batch)))

    def forward(self, input: Tensor) -> Tensor:
        if self.first:
            return torch.einsum('...j,kjb->...kb', input, self.weight) + self.bias
        else:
            return torch.einsum('...jb,kjb->...kb', input, self.weight) + self.bias


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.net2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], -1)
        return torch.cat([self.net1(xu), self.net2(xu)], -1)


class ProbablisticPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space,
                 min_log_std=np.log(1e-2)):
        super(ProbablisticPolicy, self).__init__()
        self.min_log_std = torch.tensor(min_log_std, dtype=torch.float32)
        self.factor = 2
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.factor * hidden_dim)
        self.linear3 = nn.Linear(self.factor * hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = soft_clamp(log_std, _min=LOG_SIG_MIN, _max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, num=None, det=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if not det:
            if num is None:
                x_t = normal.rsample()
            else:
                x_t = normal.rsample([num])
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= self.action_scale.log() + 2. * (np.log(2.) - x_t - F.softplus(-2. * x_t))
            log_prob = log_prob.sum(-1)
        else:
            y_t = torch.tanh(mean)
            action = y_t * self.action_scale + self.action_bias
            log_prob = None
        return action, log_prob

    def tanh_normal_sample(self, action, std, num=None):
        n_action = torch.clip((action - self.action_bias) / self.action_scale, -1. + epsilon, 1. - epsilon)
        r_action = torch.atanh(n_action)
        normal = Normal(r_action, std)
        if num is None:
            r_sample = normal.rsample()
        else:
            r_sample = normal.rsample([num])
        n_sample = torch.tanh(r_sample)
        sample = n_sample * self.action_scale + self.action_bias
        log_prob = normal.log_prob(r_sample)
        log_prob -= self.action_scale.log() + 2. * (np.log(2.) - r_sample - F.softplus(-2. * r_sample))
        log_prob = log_prob.sum(-1)
        return sample, log_prob
    
    def cal_prob(self, action, state, beta=0.):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        n_action = torch.clip((action - self.action_bias) / self.action_scale, -1.+epsilon, 1.-epsilon)
        r_action = torch.atanh(n_action)
        log_prob = normal.log_prob(r_action)
        log_prob -= self.action_scale.log() + 2. * (np.log(2.) - r_action - F.softplus(-2. * r_action))
        weight = std.detach() ** (2 * beta)
        weight /= weight.mean(-2, keepdim=True)
        log_prob *= weight
        log_prob = log_prob.sum(-1)
        return log_prob

    def to(self, device):
        self.min_log_std = self.min_log_std.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(ProbablisticPolicy, self).to(device)
    
    def get_param_values(self):
        params = torch.nn.utils.parameters_to_vector(self.parameters())
        return params
    
    def set_param_values(self, given_parameters):
        torch.nn.utils.vector_to_parameters(given_parameters, self.parameters())

    def get_grads(self):
        grads = []
        self.grads_size = []
        for param in self.parameters():
            current_grads = param.grad.data.view(-1)
            grads.append(current_grads)
            self.grads_size.append(current_grads.size(0))

        grads = torch.cat(grads)
        for param in self.parameters():
            param.grad.detach_()
            param.grad.zero_()
        return grads

    def set_grads(self, grads):
        splited_grads = torch.split(-grads, self.grads_size)
        ind = 0
        for param in self.parameters():
            grads = splited_grads[ind]
            grads = grads.view(param.size())
            param.grad.data = grads
            ind += 1

