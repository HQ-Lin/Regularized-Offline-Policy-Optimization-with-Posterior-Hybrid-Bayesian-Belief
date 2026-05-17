import torch
import numpy as np
from torch.nn import SiLU

from utilities.utils import soft_clamp


class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))

    def forward(self, x, indexes=None):
        if indexes is not None:
            weight = self.weight[indexes]
            bias = self.bias[indexes]
        else:
            weight = self.weight
            bias = self.bias
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)
        x = x + bias
        return x
    

class EnsembleRegLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, predict_dim, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features + predict_dim * 2, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        torch.nn.init.trunc_normal_(self.weight, std=1 / (2 * in_features ** 0.5))

    def forward(self, x, predicted_mu, predicted_std, indexes=None):
        if indexes is not None:
            weight = self.weight[indexes]
            bias = self.bias[indexes]
        else:
            weight = self.weight
            bias = self.bias
        x = torch.cat([x, predicted_mu, predicted_std], dim=-1)
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)
        x = x + bias
        return x

class EnsembleEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, ensemble_size=7):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.ensemble_size = ensemble_size
        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, num_embeddings, embedding_dim)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, embedding_dim)))
        torch.nn.init.trunc_normal_(self.weight, std=1 / (2 * embedding_dim ** 0.5))
    
    def forward(self, x, indexes=None):
        if indexes is not None:
            weight = self.weight[indexes]
            bias = self.bias[indexes]
        else:
            weight = self.weight
            bias = self.bias
        
        one_hot_x = torch.nn.functional.one_hot(x, num_classes=self.num_embeddings).float()
        embeddings = torch.einsum('ij,bjk->bik', one_hot_x, weight)
        embeddings = embeddings + bias
        return embeddings

class EnsembleTransition(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local', predict_reward=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.mode = mode
        self.predict_reward = predict_reward
        self.ensemble_size = ensemble_size
        self.ensemble_idxs = np.arange(ensemble_size)
        self.register_parameter('input_mean', torch.nn.Parameter(torch.zeros(obs_dim+action_dim), requires_grad=False))
        self.register_parameter('input_std', torch.nn.Parameter(torch.ones(obs_dim+action_dim), requires_grad=False))
        self.register_parameter('output_std', torch.nn.Parameter(torch.ones(obs_dim+predict_reward), requires_grad=False))
        self.register_parameter('pred_min', torch.nn.Parameter(torch.ones(obs_dim), requires_grad=False))
        self.register_parameter('pred_max', torch.nn.Parameter(torch.ones(obs_dim), requires_grad=False))
        self.register_parameter('pred_min_2', torch.nn.Parameter(torch.ones(obs_dim), requires_grad=False))
        self.register_parameter('pred_max_2', torch.nn.Parameter(torch.ones(obs_dim), requires_grad=False))
        self.reward_shift = 0.
        self.reward_scale = 1.
        self.activation = SiLU()
        self.factor_layer_size_1 = 2
        self.factor_layer_size_2 = 2

        assert hidden_layers >= 4, "The hidden layers must greater than or equal to 4 !"
        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            elif i == 1:
                module_list.append(EnsembleLinear(hidden_features, hidden_features * self.factor_layer_size_1, ensemble_size))
            elif i == 2:
                module_list.append(EnsembleLinear(hidden_features * self.factor_layer_size_1, hidden_features * self.factor_layer_size_2, ensemble_size))
            elif i == 3:
                module_list.append(EnsembleLinear(hidden_features * self.factor_layer_size_2, hidden_features, ensemble_size))   
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)
        # output mean and std
        self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.predict_reward), ensemble_size)

        self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.predict_reward) * 1, requires_grad=True))
        self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.predict_reward) * -5, requires_grad=True))

    def forward(self, obs_action, indexes=None):
        output = (obs_action - self.input_mean) / self.input_std
        for layer in self.backbones:
            output = self.activation(layer(output, indexes))
        mu, logstd = torch.chunk(self.output_layer(output, indexes), 2, dim=-1)
        
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        mu = mu * self.output_std
        logstd = logstd + self.output_std.log()

        if self.mode == 'local':
            if self.predict_reward:
                obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + obs_action[..., :self.obs_dim]
        return torch.distributions.Normal(mu, torch.exp(logstd))

    def sample_forward(self, obs_action, num=1, idx=None):
        if idx is None:
            indexes = np.random.choice(self.ensemble_idxs, num)
            indexes, counts = np.unique(indexes, return_counts=True)
            return self.forward(obs_action, indexes), counts
        else:
            return self.forward(obs_action, self.ensemble_idxs[idx])

    def reset_ensemble_size(self, ensemble_size):
        self.ensemble_idxs = np.random.choice(self.ensemble_size, ensemble_size, replace=False)
        self.ensemble_size = ensemble_size

    def reset_statistics(self, input_mean, input_std, output_std,
                         pred_min, pred_max, pred_min_2, pred_max_2,
                         reward_shift=None, reward_scale=None):
        self.input_mean.data = input_mean.clone().detach()

        idx = input_std < 1e-12
        self.input_std.data[idx] = torch.tensor(1., device=self.input_std.device)
        self.input_std.data[~idx] = input_std[~idx].clone().detach()

        idx = output_std < 1e-12
        self.output_std.data[idx] = torch.tensor(1., device=self.output_std.device)
        self.output_std.data[~idx] = output_std[~idx].clone().detach()

        self.pred_min.data = pred_min.clone().detach()
        self.pred_max.data = pred_max.clone().detach()

        self.pred_min_2.data = pred_min_2.clone().detach()
        self.pred_max_2.data = pred_max_2.clone().detach()

        if reward_shift is not None:
            self.reward_shift = reward_shift
        if reward_scale is not None:
            self.reward_scale = reward_scale

    def get_normalized_reward(self, reward):
        if not self.predict_reward:
            return reward
        else:
            return (reward - self.reward_shift) / self.reward_scale

    def get_raw_reward(self, reward):
        if not self.predict_reward:
            return reward
        else:
            return reward * self.reward_scale + self.reward_shift

class EnsembleRegression(EnsembleTransition):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local', predict_reward=True):
        super(EnsembleRegression, self).__init__(obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size, mode, predict_reward)
        self.embedding = EnsembleEmbedding(obs_dim + self.predict_reward, hidden_features, ensemble_size)
        
        assert hidden_layers >= 4, "The hidden layers must greater than or equal to 4 !"
        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleRegLinear(obs_dim + action_dim, hidden_features, obs_dim + self.predict_reward, ensemble_size))
            elif i == 1:
                module_list.append(EnsembleRegLinear(hidden_features, hidden_features * 2, obs_dim + self.predict_reward, ensemble_size))
            elif i == 2:
                module_list.append(EnsembleRegLinear(hidden_features * 2, hidden_features * 4, obs_dim + self.predict_reward, ensemble_size))
            elif i == 3:
                module_list.append(EnsembleRegLinear(hidden_features * 4, hidden_features, obs_dim + self.predict_reward, ensemble_size))   
            else:
                module_list.append(EnsembleRegLinear(hidden_features, hidden_features, obs_dim + self.predict_reward, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        self.output_layer = EnsembleRegLinear(hidden_features, 2, obs_dim + self.predict_reward, ensemble_size)

    
    def forward(self, obs_action, indexes=None):
        output = (obs_action - self.input_mean) / self.input_std
        if len(output.shape) == 2:
            output = output.unsqueeze(0).expand(indexes.shape[0], -1, -1)
            predicted_mu = torch.zeros((output.shape[0], output.shape[1], self.obs_dim + 1), device=output.device).requires_grad_(False)
            predicted_std = torch.zeros((output.shape[0], output.shape[1], self.obs_dim + 1), device=output.device).requires_grad_(False)
        else:
            predicted_mu = torch.zeros((output.shape[0], output.shape[1], self.obs_dim + 1), device=output.device).requires_grad_()
            predicted_std = torch.zeros((output.shape[0], output.shape[1], self.obs_dim + 1), device=output.device).requires_grad_()
        
        obs_act_dim = output.shape[1]
        for times in range(self.obs_dim + self.predict_reward):
            flag = True
            for layer in self.backbones:
                if flag:
                    t = torch.tensor(times, dtype=torch.long, device=output.device).expand(obs_act_dim)
                    output_fi = self.activation(layer(output, predicted_mu, predicted_std, indexes) + self.embedding(t, indexes))
                    flag = False
                else:
                    output_fi = self.activation(layer(output_fi, predicted_mu, predicted_std, indexes))
            mu, logstd = torch.chunk(self.output_layer(output_fi, predicted_mu, predicted_std, indexes), 2, dim=-1)
            
            predicted_mu = torch.cat([predicted_mu[..., :times],
                                    mu,
                                    predicted_mu[..., times + 1:]
                                    ], dim=-1)
            predicted_std = torch.cat([predicted_std[..., :times],
                                    logstd,
                                    predicted_std[..., times + 1:]
                                    ], dim=-1)
            
        predicted_std = soft_clamp(predicted_std, self.min_logstd, self.max_logstd)
        predicted_mu = predicted_mu * self.output_std
        predicted_std = predicted_std + self.output_std.log()

        if self.mode == 'local':
            if self.predict_reward:
                obs, reward = torch.split(predicted_mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                predicted_mu = torch.cat([obs, reward], dim=-1)
            else:
                predicted_mu = predicted_mu + obs_action[..., :self.obs_dim]
        
        return torch.distributions.Normal(predicted_mu, torch.exp(predicted_std))