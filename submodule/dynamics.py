import torch
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F 

BATCH_SIZE_T = None
BATCH_SIZE_A = None


class PseudoDynamics(object):
    def __init__(self, state_buffer, agent, transition, args, done_func=None, reward_func=None):
        self.state_buffer = state_buffer
        self.device = args.device
        self.agent = agent
        self.transition = transition
        if reward_func is not None:
            self.reward_func = reward_func
        else:
            self.reward_func = None
        self.done_func = done_func

        self.use_adaptive_belief = args.use_adaptive_belief
        self.adv_batch_size = args.adv_batch_size
        self.num_sample_transition = args.num_sample_transition
        self.order_transition = args.order_transition
        self.gamma = args.gamma
        self.MC_size_state = args.MC_size_state
        self.MC_size_action = args.MC_size_action
        self.adv_horizon = args.adv_horizon
        self.adv_explore_ratio = args.adv_explore_ratio
        self.reweight_explore_ratio = args.reweight_explore_ratio
        self.dynamics_reweight = args.dynamics_reweight
        self.k_smallest_ensemble = args.topk
        self.ablation_belief = args.ablation_belief

        self.belief_history_log = []
        self.surprise_history_log = []
        self.shift_min = (self.transition.pred_min + self.transition.pred_min_2) / 2.
        self.scale_min = 12. / (self.transition.pred_min - self.transition.pred_min_2)
        self.shift_max = (self.transition.pred_max + self.transition.pred_max_2) / 2.
        self.scale_max = 12. / (self.transition.pred_max - self.transition.pred_max_2)
        self.out_scale = 1. / torch.sigmoid(torch.tensor(6., device=self.shift_min.device))

        assert not (self.dynamics_reweight and self.use_adaptive_belief)

        self.reset()

    def reset(self, state=None):
        if state is None:
            sample_idx = np.random.choice(len(self.state_buffer), self.adv_batch_size)
            self.state = torch.as_tensor(self.state_buffer[sample_idx], device=self.device)
            # self.state.shape [adv_batch_size, state_dim]
        else:
            self.state = torch.as_tensor(state, device=self.device)
        
        if self.use_adaptive_belief:
            self.belief_b = torch.ones(
                self.state.shape[0], self.num_sample_transition, device=self.device
            ) / self.num_sample_transition
        self.belief_history_log.clear()
        self.surprise_history_log.clear()
        
        self.num_steps = torch.zeros(self.state.shape[0], device=self.device)
        return self.state.cpu().numpy()

    def step(self, action, test_mode=False):
        action_torch = torch.as_tensor(action, device=self.device)
 
        if np.random.uniform() < self.reweight_explore_ratio or not self.dynamics_reweight:

            adv_q_target, adv_q, next_state, reward, done_prob, ensemble_std = \
                self._pseudo_q_value_estimates(self.state, action_torch, test_mode)
        else:
            adv_q_target, adv_q, next_state, reward, done_prob, ensemble_std = \
                self._reweight_predict_q(self.state, action_torch, test_mode)
            
        rollout_next_state = next_state[0]
        rollout_reward = reward[0]
        rollout_done = torch.bernoulli(done_prob[0]).type(torch.bool)
        self.num_steps += 1
        terminal = torch.logical_or(rollout_done, self.num_steps > self.adv_horizon)
        if not test_mode:
            self.state[~terminal] = rollout_next_state[~terminal]
            # Check for termination samples
            if terminal.sum() > 0:
                sample_idx = np.random.choice(len(self.state_buffer), terminal.sum().item())
                self.state[terminal] = torch.as_tensor(self.state_buffer[sample_idx], device=self.device)
                self.num_steps[terminal] = 0

                if self.use_adaptive_belief:
                    self.belief_b[terminal] = 1. / self.num_sample_transition
        else:
            self.state = rollout_next_state[~terminal]
            self.num_steps = self.num_steps[~terminal]

            if self.use_adaptive_belief:
                self.belief_b = self.belief_b[~terminal]
            
        return (rollout_reward.cpu().numpy(), rollout_next_state.cpu().numpy(), rollout_done.cpu().numpy()), \
               adv_q_target.cpu().numpy(), adv_q.cpu().numpy(), \
               ensemble_std.cpu().numpy() if ensemble_std is not None else None

    def _transition_predict(self, state, action, test_mode):
        data_size = state.shape[0]
        if BATCH_SIZE_T is None:
            batch_size = data_size
        else:
            batch_size = BATCH_SIZE_T
        batch_num = int(np.ceil(data_size / batch_size))

        s_a = torch.cat([state, action], dim=-1)
        allow_replace = True if not self.dynamics_reweight and not self.use_adaptive_belief else False
        next_state_mean_reg = []
        next_state_std_reg = []
        # choose self.num_sample_transition integers from [0, self.transition.ensemble_size - 1]
        indexes = np.random.choice(self.transition.ensemble_size, self.num_sample_transition, replace=allow_replace)
        indexes, counts = np.unique(indexes, return_counts=True)
        # [len(indexes), adv_batch_size]
        counts_repeat = np.expand_dims(counts, axis=1).repeat(s_a.shape[0], axis=1)

        for i_batch in range(batch_num):
            s_a_batch = s_a[i_batch * batch_size: (i_batch + 1) * batch_size]
            output_batch = self.transition.sample_forward(s_a_batch, idx=indexes)
            # output_batch.mean.shape: [len(indexes), adv_batch_size, next_state_dim + 1]
            next_state_mean_reg.append(output_batch.mean)
            next_state_std_reg.append(output_batch.stddev)
        next_state_mean = torch.cat(next_state_mean_reg, dim=1)
        next_state_std = torch.cat(next_state_std_reg, dim=1)
        if test_mode:
            next_state_mean_all = next_state_mean.repeat_interleave(torch.tensor(counts, device=next_state_mean.device),
                                                                    dim=0)
            ensemble_std = next_state_mean_all.var(0).sum(-1) ** 0.5
        else:
            ensemble_std = None
        return Normal(next_state_mean, next_state_std), counts_repeat, ensemble_std

    def _predict_v(self, state):
        data_size = state.shape[2]
        if BATCH_SIZE_A is None:
            batch_size = data_size
        else:
            batch_size = BATCH_SIZE_A
        batch_num = int(np.ceil(data_size / batch_size))

        v_target_reg = []
        v_reg = []
        for i_batch in range(batch_num):
            state_batch = state[:, :, i_batch * batch_size: (i_batch + 1) * batch_size]
            v_target_batch, v_batch = self.agent.value_estimates(state_batch,
                                                            MC_size_action=self.MC_size_action,
                                                            with_RT_critic=True,
                                                            clip_Q=self.reward_func is None)
            v_target_reg.append(v_target_batch)
            v_reg.append(v_batch)
        v_target = torch.cat(v_target_reg, dim=2)
        v = torch.cat(v_reg, dim=2)
        return v_target, v

    def _q_value_estimates(self, state, action, next_state, next_value_target, next_value, mean_reward):
        tile_size = np.prod(next_state.shape[0:2])
        state_ = state.repeat([tile_size, 1])
        action_ = action.repeat([tile_size, 1])
        next_state_ = next_state.reshape([-1, next_state.shape[-1]])
        mean_reward_ = mean_reward.reshape([-1, 1])
        reward_, done_prob_ = self._get_reward_done(state_, action_, next_state_, mean_reward_)
        q_value_target_ = reward_.reshape([-1]) + \
                          (1. - done_prob_).reshape([-1]) * self.gamma * next_value_target.reshape([-1])
        q_value_ = reward_.reshape([-1]) + \
                   (1. - done_prob_).reshape([-1]) * self.gamma * next_value.reshape([-1])
        return q_value_target_.reshape_as(next_value_target), \
               q_value_.reshape_as(next_value), \
               reward_.reshape_as(next_value), \
               done_prob_.reshape_as(next_value)

    def _reweight_predict_q(self, state, action, test_mode):
        prob_trans, counts, ensemble_std = self._transition_predict(state, action, test_mode)
        if prob_trans.mean.shape[-1] > state.shape[-1]:
            mean_trans = prob_trans.mean
            std_trans = prob_trans.stddev
            prob_next_state = Normal(mean_trans[..., :-1], std_trans[..., :-1])
            mean_reward = mean_trans[..., -1].expand([self.MC_size_state, -1, -1])
        else:
            prob_next_state = prob_trans
            mean_reward = None
        next_state_candi = prob_next_state.sample([self.MC_size_state])
        next_value_target, next_value = self._predict_v(next_state_candi)

        q_value_target, q_value, reward_candi, done_prob_candi \
            = self._q_value_estimates(state, action, next_state_candi, next_value_target, 
                              next_value, mean_reward)
        
        num_ensemble_models = q_value.shape[1] 
        batch = q_value.shape[-1]
        assert self.k_smallest_ensemble <= num_ensemble_models
        # [len(indexes), adv_batch_size] 
        mean_q_ensemble = q_value.mean(dim=0)
        lambda_ = -3.0
        smallest_k_q_values, smallest_k_indices = torch.topk(mean_q_ensemble, self.k_smallest_ensemble,
                                
        min_k_q_per_item = smallest_k_q_values.min(dim=0, keepdim=True).values 
        max_k_q_per_item = smallest_k_q_values.max(dim=0, keepdim=True).values 

        delta_q_per_item = max_k_q_per_item - min_k_q_per_item

        normalized_smallest_k_q = torch.where(
                delta_q_per_item.abs() < 1e-8,
                torch.zeros_like(smallest_k_q_values), # True 
                (smallest_k_q_values - min_k_q_per_item) / (delta_q_per_item + 1e-8) 
            )

        sub_weights = F.softmax(lambda_ * normalized_smallest_k_q, dim=0)
        current_weights = torch.zeros_like(mean_q_ensemble)
        # [len(indexes), adv_batch_size]
        current_weights.scatter_(dim=0, index=smallest_k_indices, src=sub_weights)

        unif = 0.01
        weights = (1. - unif) * current_weights + unif / self.num_sample_transition
        weights_expanded = weights.unsqueeze(0).expand(
            self.MC_size_state, num_ensemble_models, batch
        )

        def compute_weighted_final(variable):
            weighted_var = (weights_expanded * variable).sum(dim=1)
            return weighted_var
        # [adv_batch_size]
        final_q = compute_weighted_final(q_value)[0]
        final_q_target = compute_weighted_final(q_value_target)[0]
        final_reward = compute_weighted_final(reward_candi)
        final_done_prob = compute_weighted_final(done_prob_candi)
        weights_expanded_state = weights_expanded.unsqueeze(-1)
        final_next_state = (weights_expanded_state * next_state_candi).sum(dim=1)  
        return final_q_target, final_q, final_next_state, final_reward, final_done_prob, ensemble_std

    def _pseudo_q_value_estimates(self, state, action, test_mode):
        prob_trans, counts, ensemble_std = self._transition_predict(state, action, test_mode)
        if prob_trans.mean.shape[-1] > state.shape[-1]:
            mean_trans = prob_trans.mean
            std_trans = prob_trans.stddev
            prob_next_state = Normal(mean_trans[..., :-1], std_trans[..., :-1])
            mean_reward = mean_trans[..., -1].expand([self.MC_size_state, -1, -1])
        else:
            prob_next_state = prob_trans
            mean_reward = None
        next_state_candi = prob_next_state.sample([self.MC_size_state])
        next_value_target, next_value = self._predict_v(next_state_candi)

        q_value_target, q_value, reward_candi, done_prob_candi \
            = self._q_value_estimates(state, action, next_state_candi, next_value_target, next_value, mean_reward)
        
        q_value_target = q_value_target.mean(0)
        q_value = q_value.mean(0)
        counts = torch.as_tensor(counts, device=self.device)

        _, adv_q_target = self._select_dynamics(q_value_target, counts)

        if np.random.uniform() < self.adv_explore_ratio and not test_mode:
            idx_transition = torch.multinomial(counts.to(torch.float32).t(), num_samples=1)
            adv_q = q_value.gather(dim=0, index=idx_transition.reshape(1, -1)).squeeze(0)
        else:
            idx_transition, adv_q = self._select_dynamics(q_value, counts)

        idx_gather = idx_transition.reshape(-1, 1).expand(next_state_candi.shape[0], 1, -1, next_state_candi.shape[-1])
        next_state = next_state_candi.gather(dim=1, index=idx_gather).squeeze(dim=1)
        idx_gather = idx_transition.reshape(-1).expand(reward_candi.shape[0], 1, -1)
        reward = reward_candi.gather(dim=1, index=idx_gather).squeeze(dim=1)
        done_prob = done_prob_candi.gather(dim=1, index=idx_gather).squeeze(dim=1)
        return adv_q_target, adv_q, next_state, reward, done_prob, ensemble_std


    def _select_dynamics(self, q_value, counts):
        idx_sorted = torch.argsort(q_value, dim=0)
        counts_sorted = counts.gather(dim=0, index=idx_sorted)
        cumsum_counts = counts_sorted.cumsum(dim=0)
        
        idx = torch.argmax((cumsum_counts >= self.order_transition) * 1., dim=0, keepdim=True)
        idx_transition = idx_sorted.gather(dim=0, index=idx)
        ad_q_value = q_value.gather(dim=0, index=idx_transition)
        return idx_transition.squeeze(0), ad_q_value.squeeze(0)
        

    def _get_reward_done(self, state, action, next_state, mean_reward):
        if self.reward_func is not None:
            data_ = {"obs": state.cpu().numpy(),
                     "action": action.cpu().numpy(),
                     "next_obs": next_state.cpu().numpy()}
            reward = self.reward_func(data_)
            reward = torch.tensor(reward, device=self.device)
        else:
            reward = mean_reward
            reward = torch.clip(reward, min=0., max=1.)

        done = self.done_func(state.cpu().numpy(),
                              action.cpu().numpy(),
                              next_state.cpu().numpy())
        done = torch.tensor(done, device=self.device, dtype=torch.float32)

        continue_prob = torch.clip(torch.minimum(torch.sigmoid((next_state - self.shift_min) * self.scale_min),
                                                  torch.sigmoid((next_state - self.shift_max) * self.scale_max))
                                   * self.out_scale, max=1.).prod(-1, keepdim=True)
        done_prob = torch.maximum(done, 1. - continue_prob).clamp(5e-7, 1.0 - 5e-7)
        return reward, done_prob

    def get_belief_history(self):
        """
        Convert the recorded belief log into a NumPy array.
        
        return:
            np.ndarray:  [num_steps, adv_batch_size, num_ensemble_models]
        """
        if not self.belief_history_log:
            return np.array([])
        return np.stack(self.belief_history_log, axis=0)
    
    def get_surprise_history(self):
        """
        Convert the recorded belief log into a NumPy array.
        
        return:
            np.ndarray:  [num_steps, num_ensemble_models, adv_batch_size]
        """
        if not self.surprise_history_log:
            return np.array([])
        return np.stack(self.surprise_history_log, axis=0)