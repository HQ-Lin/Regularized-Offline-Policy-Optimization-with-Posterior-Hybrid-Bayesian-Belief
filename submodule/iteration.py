import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from torch.optim import Adam
from optim.optimizer import SGD, RAD, RGD, NAG
from utilities.utils import soft_update, hard_update
from layer.policy import ProbablisticPolicy, QNetwork

REAL_RATIO = 0.5

def cal_alpha(target_kld, qf, log_ratio=None, min_val=1e-3):
    if log_ratio is None:
        return torch.clip(
            (
                (qf * (qf / min_val).softmax(dim=0)).sum(dim=0)
                - qf.mean(dim=0)
            ) / target_kld,
            min=min_val
        )
    else:
        return torch.clip(
            (
                (qf * (log_ratio + qf / min_val).softmax(dim=0)).sum(dim=0)
                - (log_ratio.softmax(dim=0) * qf).sum(dim=0)
            ) / target_kld,
            min=min_val
        )
    


class PolicyIteration(object):
    def __init__(self, state_dim, action_space, real_memory, real_seq_memory, args, value_clip):
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.target_kld = args.target_kld
        self.automatic_alpha_tuning = args.automatic_alpha_tuning
        self.real_batch_size = args.real_batch_size
        self.MC_size_action = args.MC_size_action
        self.explore_ratio = args.explore_ratio
        self.real_ratio = args.real_ratio

        self.value_clip = value_clip
        self.tau_value = args.tau_value
        self.tau_policy = args.tau_policy

        self.policy_type = args.policy_type
        self.det_policy = args.det_policy

        self.device = args.device

        self.use_bc_regularization = args.use_bc_regularization
        self.bc_weight = args.bc_weight

        self.bregman_reg = args.bregman_reg
        self.lamba = args.lamba
        self.divergence_decay = args.beta
        self.initialize_breg_grad = True
        self.max_grad = args.L2_max_grad
        self.steps = 0
        self.proj_lr = args.proj_lr
        self.breg_lr = args.actor_lr
        self.grad_factor = args.grad_factor
        self.momentum = args.beta

        self.real_memory = real_memory
        self.real_seq_memory = real_seq_memory
        self.args = args
        self.task = args.task

        self.critic = QNetwork(state_dim, action_space.shape[0], args.agent_layer_size).to(device=self.device)
        if self.policy_type == "Gaussian":
            self.policy_ref = ProbablisticPolicy(state_dim, action_space.shape[0], args.agent_layer_size, action_space).to(
                self.device)
            self.policy = ProbablisticPolicy(state_dim, action_space.shape[0], args.agent_layer_size, action_space).to(
                self.device)
        else:
            quit('Policy type is not supported.')
        hard_update(self.policy_ref, self.policy)
        self.optim = args.optim
        if self.optim == "Adam":
            self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.actor_lr)
        elif self.optim == "SGD":
            self.critic_optim = SGD(self.critic.parameters(), lr=args.critic_lr, momentum=0.9, output_info=False)
            self.policy_optim = SGD(self.policy.parameters(), lr=args.actor_lr, momentum=0.9, output_info=False)
        elif self.optim == "RAD":
            self.critic_optim = RAD(self.critic.parameters(), lr=args.critic_lr, output_info=False)
            self.policy_optim = RAD(self.policy.parameters(), lr=args.actor_lr, output_info=False)
        elif self.optim == "RGD":
            self.critic_optim = RGD(self.critic.parameters(), lr=args.critic_lr, output_info=False)
            self.policy_optim = RGD(self.policy.parameters(), lr=args.actor_lr, output_info=False)
        elif self.optim == "NAG":
            self.critic_optim = NAG(self.critic.parameters(), lr=args.critic_lr, momentum=1e-5, output_info=False)
            self.policy_optim = NAG(self.policy.parameters(), lr=args.actor_lr, momentum=1e-5, output_info=False)

        self.critic_target = QNetwork(state_dim, action_space.shape[0], args.agent_layer_size).to(self.device)
        hard_update(self.critic_target, self.critic)


    def diag_projection(self, x_t, t, grad, F):
        out = x_t - t * grad / (F.sqrt() + 1e-8)
        return out
    
    def kl_projection(self, x_t, t, grad):
        sign_x = torch.sign(x_t)
        abs_x = torch.abs(x_t)
        simplex_x = abs_x / abs_x.sum()

        
        simplex_grad = sign_x * grad
        
        x_t1 = simplex_x * torch.exp(-t * simplex_grad)
        x_t1 = x_t1/x_t1.sum()

        x_t1 = x_t1 * sign_x * abs_x.sum()
        return x_t1
    
    @torch.no_grad()
    def _compute_n_step_targets_from_real_data(self, 
                                            n_step: int):

        s_h, a_h, rewards_n, dones_n, s_h_plus_n = \
            self.real_seq_memory.sample_sequential_transitions(
                self.real_batch_size, n_step
            )
        

        s_h = torch.as_tensor(s_h, device=self.device)
        a_h = torch.as_tensor(a_h, device=self.device)
        rewards_n = torch.as_tensor(rewards_n, device=self.device)
        dones_n = torch.as_tensor(dones_n, device=self.device, dtype=torch.float32)
        s_h_plus_n = torch.as_tensor(s_h_plus_n, device=self.device)

        next_value = self.value_estimates(s_h_plus_n, 
                                    MC_size_action=self.MC_size_action, 
                                    clip_Q=self.value_clip) 

        
        G = next_value # (B,)
        
        for t in reversed(range(n_step)):
 
            r_t = rewards_n[:, t, :].squeeze(-1) 
            d_t = dones_n[:, t, :].squeeze(-1)   
            G = r_t + self.gamma * (1. - d_t) * G
        

        return s_h, a_h, G

    def value_estimates(self, state, MC_size_action=10, with_RT_critic=False, clip_Q=True):
        state_torch = torch.as_tensor(state, device=self.device)
        with torch.no_grad():
            if self.policy_type in ["Gaussian"]:
                if self.det_policy:
                    action, _ = self.policy_ref.sample(state_torch, det=True)
                    qf_target = self.critic_target(state_torch, action)
                    qf_target = qf_target.min(-1)[0]
                    if clip_Q:
                        qf_target = torch.clip(qf_target, 0., 1 / (1 - self.gamma))
                    if with_RT_critic:
                        qf = self.critic(state_torch, action)
                        qf = qf.min(-1)[0]
                        if clip_Q:
                            qf = torch.clip(qf, 0., 1 / (1 - self.gamma))
                        return qf_target, qf
                    else:
                        return qf_target
                else:
                    IS_size = int(MC_size_action // 2)

                    # To compute the expectation under a reference policy,
                    # but employ importance sampling using the current policy for sampling.
                    action_ref, log_prob_ref = self.policy_ref.sample(state_torch, num=MC_size_action - IS_size)
                    # current policy
                    # adv_iter [IS_size, MC_size_state, len(indexes), adv_batch_size, action_dim]
                    # real_iter [IS_size, real_batch_size, action_dim]
                    action_proposal, log_prob_proposal = self.policy.sample(state_torch, num=IS_size)
                    action = torch.cat([action_proposal, action_ref], dim=0)
                    
                    # Calculate the log probability of the action_proposal under state_torch.
                    # adv shape [IS_size, MC_size_state, len(indexes), adv_batch]
                    # real shape [IS_size, real_batch]
                    log_prob_IS = self.policy_ref.cal_prob(action_proposal, state_torch)

                    # adv shape [MC_size_action, MC_size_state, len(indexes), adv_batch]
                    # real shape [MC_size_action, real_batch]
                    log_IS_ratio = torch.cat([log_prob_IS - log_prob_proposal, log_prob_ref - log_prob_ref], dim=0)

                    # adv shape [MC_size_action, MC_size_state, len(indexes), adv_batch, state_dim]
                    # real shape [MC_size_action, real_batch, state_dim]
                    state_repeated = state_torch.expand([MC_size_action, *([-1] * state_torch.ndim)])

                    # adv shape [MC_size_action, MC_size_state, len(indexes), adv_batch, 2]
                    # real shape [MC_size_action, real_batch, 2]
                    qf_target = self.critic_target(state_repeated, action)

                    # adv shape [MC_size_action, MC_size_state, len(indexes), adv_batch]
                    # real shape [MC_size_action, real_batch]
                    qf_target = qf_target.min(-1)[0]
                    if clip_Q:
                        qf_target = torch.clip(qf_target, 0., 1 / (1 - self.gamma))

                    alpha = cal_alpha(self.target_kld, qf_target, log_IS_ratio) if self.automatic_alpha_tuning \
                        else self.alpha
                    qf_alpha_target = qf_target / alpha
                    qf_alpha_target += log_IS_ratio

                    # adv shape [MC_size_state, len(indexes), adv_batch]
                    # real shape [real_batch]
                    # minus log_IS_ratio.logsumexp(dim=0) is to normalize
                    v_target = alpha * (qf_alpha_target.logsumexp(dim=0) - log_IS_ratio.logsumexp(dim=0))

                    if with_RT_critic:
                        qf = self.critic(state_repeated, action)
                        qf = qf.min(-1)[0]
                        if clip_Q:
                            qf = torch.clip(qf, 0., 1 / (1 - self.gamma))

                        alpha = cal_alpha(self.target_kld, qf, log_IS_ratio) if self.automatic_alpha_tuning \
                            else self.alpha
                        qf_alpha = qf / alpha
                        qf_alpha += log_IS_ratio
                        v = alpha * (qf_alpha.logsumexp(dim=0) - log_IS_ratio.logsumexp(dim=0))
                        return v_target, v
                    else:
                        return v_target
            else:
                quit('Policy type is not supported.')

    def offline_update_parameters(self, adv_state, adv_action, adv_q):
        # below two [adv_batch_size, dim]
        adv_state_torch = torch.as_tensor(adv_state, device=self.device)
        adv_action_torch = torch.as_tensor(adv_action, device=self.device)
        # adv_q_torch [adv_batch_size]
        adv_q_torch = torch.as_tensor(adv_q, device=self.device)

        if "antmaze" in self.task:
            N_STEP = self.args.n_step
            real_state_torch, real_action_torch, real_q_value = self._compute_n_step_targets_from_real_data(N_STEP)
        else:
            # state & next_state [real_batch_size, state_dim]
            # action & next_action [real_batch_size, action_dim]
            # reward & done [real_batch_size]
            real_state, real_action, real_reward, real_next_state, real_done = self.real_memory.sample(self.real_batch_size)
            real_state_torch = torch.as_tensor(real_state, device=self.device)
            real_next_state_torch = torch.as_tensor(real_next_state, device=self.device)
            real_action_torch = torch.as_tensor(real_action, device=self.device)
            real_reward_torch = torch.as_tensor(real_reward, device=self.device).squeeze(1)
            real_done_torch = torch.as_tensor(real_done, dtype=torch.float32, device=self.device).squeeze(1)
            
            # v & real_q_value [real_batch_size]
            v = self.value_estimates(real_next_state_torch, MC_size_action=self.MC_size_action, clip_Q=self.value_clip)
            real_q_value = real_reward_torch + (1. - real_done_torch) * self.gamma * v
            
        # state & action [adv_batch_size + real_batch_size, dim]
        state_torch = torch.cat([real_state_torch, adv_state_torch], dim=0)
        action_torch = torch.cat([real_action_torch, adv_action_torch], dim=0)
        # [adv_batch_size + real_batch_size]
        q_value = torch.cat([real_q_value, adv_q_torch], dim=0)
        
        if self.value_clip:
            q_value = torch.clip(q_value, 0., 1 / (1 - self.gamma))

        # qf & q_value [adv_batch_size + real_batch_size ,2]
        qf = self.critic(state_torch, action_torch)
        q_value = q_value.repeat([2, 1]).t()
        
        qf_loss = F.mse_loss(qf[:self.real_batch_size],
                             q_value[:self.real_batch_size], reduction='none'
                             ).mean(0).sum(0) * self.real_ratio \
                  + F.mse_loss(qf[self.real_batch_size:],
                               q_value[self.real_batch_size:], reduction='none'
                               ).mean(0).sum(0) * (1. - self.real_ratio)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if self.policy_type in ["Gaussian"]:
            with torch.no_grad():
                IS_size = int(self.MC_size_action // 2)
                action_ref, log_prob_ref = self.policy_ref.sample(state_torch, num=self.MC_size_action - IS_size)
                
                # generate adv_action and real_action, to encourage exploration
                action_1, log_prob_1 = \
                    self.policy_ref.tanh_normal_sample(real_action_torch, std=0.1, num=IS_size)
                action_2, log_prob_2 = self.policy.sample(adv_state_torch, num=IS_size)
                
                action_proposal = torch.cat([action_1, action_2], dim=1)
                action = torch.cat([action_proposal, action_ref], dim=0)

                log_prob_proposal = torch.cat([log_prob_1, log_prob_2], dim=1)
                log_prob_IS = self.policy_ref.cal_prob(action_proposal, state_torch)
                log_IS_ratio = torch.cat([log_prob_IS - log_prob_proposal, log_prob_ref - log_prob_ref], dim=0)

                state_repeated = state_torch.expand([self.MC_size_action, *([-1] * state_torch.ndim)])
                qf_pi = self.critic(state_repeated, action)
                qf_pi = qf_pi.min(-1)[0]
                if self.value_clip:
                    qf_pi = torch.clip(qf_pi, 0., 1 / (1 - self.gamma))

                alpha = cal_alpha(self.target_kld, qf_pi, log_IS_ratio) if self.automatic_alpha_tuning \
                    else self.alpha
                log_weight = qf_pi / alpha + log_IS_ratio
                # [self.MC_size_action, adv_batch_size + real_batch_size]
                weight = softmax(log_weight, dim=0)
        else:
            quit('Policy type is not supported.')

        # [adv_batch_size + real_batch_size]
        policy_loss = - torch.mul(self.policy.cal_prob(action, state_torch, beta=0.5), weight).sum(0)
        policy_loss = policy_loss[:self.real_batch_size].mean() * self.real_ratio \
                      + policy_loss[self.real_batch_size:].mean() * (1. - self.real_ratio)

        # compute behavior regularization loss
        if self.use_bc_regularization:
            log_prob_bc = self.policy.cal_prob(real_action_torch, real_state_torch)
            bc_loss = -log_prob_bc.mean()
            policy_loss += self.bc_weight * bc_loss


        self.policy_optim.zero_grad()
        if self.bregman_reg:
            policy_loss *= self.lamba
        policy_loss.backward()
        self.policy_optim.step()
        self.steps += 1
        if self.bregman_reg:
            grads = self.policy.get_grads() / (self.lamba + 1e-6)
            u_buffer = torch.zeros_like(grads)
            grad_rmean = torch.zeros_like(grads)
            G_p = grads.norm(p=2)

            if G_p < self.max_grad:
                g_max = G_p.item()
            else:
                g_max = self.max_grad

            if self.initialize_breg_grad:
                u_momentum = -grads
                grad_rmean = u_momentum.pow(2)
                self.initialize_breg_grad = False
            else:
                u_momentum = self.momentum * (-grads) + (1 - self.momentum) * (u_buffer)
                grad_rmean = self.divergence_decay * grad_rmean + (1 - self.divergence_decay) * u_momentum.pow(2)

            eta_lr = max(self.breg_lr / ((1 + self.grad_factor * self.steps) ** (1 / 2)), self.breg_lr / 2)

            u_momentum = torch.clamp(u_momentum, -g_max, g_max)
            u_buffer = u_momentum.detach().clone()
            params = self.policy.get_param_values()
            hat_grad_rmean = grad_rmean / (1 - math.pow(self.divergence_decay, self.steps))
            ''' 
            params_hat = self.lamba * self.kl_projection(params, self.proj_lr, u_momentum, hat_grad_rmean) + \
                (1 - self.lamba) * self.diag_projection(params, self.proj_lr, u_momentum, hat_grad_rmean)
            '''
            params_hat = self.diag_projection(params, self.proj_lr, u_momentum, hat_grad_rmean)
            params = params + (1 - self.lamba) * eta_lr * (params_hat - params)
            self.policy.set_param_values(params)

        soft_update(self.critic_target, self.critic, self.tau_value)
        soft_update(self.policy_ref, self.policy, self.tau_policy)

        return qf_loss.item(), policy_loss.item()

    def act(self, state, mode=0):
        # [adv_batch_size, state_dim]
        state_torch = torch.as_tensor(state, device=self.device).reshape([-1, state.shape[-1]])
        with torch.no_grad():
            if mode == 0:
                # importance-sample according to policy_ref(a|s)*exp(Q(s,a)/α), with exploration following policy(a|s)
                IS_size = int(self.MC_size_action // 2)
                bool_index = np.random.uniform(size=state_torch.shape[0]) > self.explore_ratio
                candi_action_ref, log_prob_ref = self.policy_ref.sample(state_torch, num=self.MC_size_action - IS_size)
                candi_action_proposal, log_prob_proposal = self.policy.sample(state_torch, num=IS_size)
                candi_action = torch.cat([candi_action_proposal, candi_action_ref], dim=0)
                # [IS_size, adv_batch_size]
                log_prob_IS = self.policy_ref.cal_prob(candi_action_proposal, state_torch)
                # [MC_size_action, adv_batch_size]
                log_IS_ratio = torch.cat([
                                        log_prob_IS[:, bool_index] - log_prob_proposal[:, bool_index],
                                        log_prob_ref[:, bool_index] - log_prob_ref[:, bool_index]], dim=0)
                
                # [MC_size_action, adv_batch_size, state_dim]
                state_exploit = state_torch[bool_index].expand([self.MC_size_action, *([-1] * state_torch.ndim)])
                # [MC_size_action, adv_batch_size, 2]
                qf = self.critic(state_exploit, candi_action[:, bool_index])
                # [MC_size_action, adv_batch_size]
                qf = qf.min(-1)[0]

                alpha = cal_alpha(self.target_kld, qf, log_IS_ratio) if self.automatic_alpha_tuning else self.alpha
                log_weight = qf / alpha
                log_weight += log_IS_ratio

                # [adv_batch_size, action_dim]
                action = torch.empty([state_torch.shape[0], candi_action.shape[-1]], device=self.device)
                
                # [adv_batch_siz, 1]
                index = torch.multinomial(softmax(log_weight.t(), dim=1), 1)
                index_gather = index.reshape(-1, 1).expand(1, -1, candi_action.shape[-1])
                action[bool_index] = candi_action[:, bool_index].gather(dim=0, index=index_gather).squeeze(dim=0)
                action[~bool_index] = candi_action[0, ~bool_index]
            elif mode == 1:
                if self.policy_type == "Gaussian":
                    # policy_ref mean
                    action, _ = self.policy_ref.sample(state_torch, det=True)
                else:
                    # sample according to policy_ref(a|s)
                    action, _ = self.policy_ref.sample(state_torch)
            elif mode == 2:
                if self.policy_type == "Gaussian":
                    # policy mean
                    action, _ = self.policy.sample(state_torch, det=True)
                else:
                    # sample according to policy(a|s)
                    action, _ = self.policy.sample(state_torch)
            elif mode == 3:
                # sample according to policy_ref(a|s)*exp(Q(s,a)/α)
                candi_action, log_prob = self.policy_ref.sample(state_torch, num=self.MC_size_action)

                state_repeated = state_torch.expand([self.MC_size_action, *([-1] * state_torch.ndim)])
                qf = self.critic(state_repeated, candi_action)
                qf = qf.min(-1)[0]

                alpha = cal_alpha(self.target_kld, qf) if self.automatic_alpha_tuning else self.alpha

                index = torch.multinomial(softmax((qf / alpha).t(), dim=1), 1)
                index_gather = index.reshape(-1, 1).expand(1, -1, candi_action.shape[-1])
                action = candi_action.gather(dim=0, index=index_gather).squeeze(dim=0)
            elif mode == 4:
                # draw N samples according to policy_ref(a|s) and find argmax_a Q(s,a)
                candi_action, log_prob = self.policy_ref.sample(state_torch, num=self.MC_size_action)

                state_repeated = state_torch.expand([self.MC_size_action, *([-1] * state_torch.ndim)])
                qf = self.critic(state_repeated, candi_action)
                qf = qf.min(-1)[0]

                index = torch.argmax(qf, dim=0)
                index_gather = index.reshape(-1, 1).expand(1, -1, candi_action.shape[-1])
                action = candi_action.gather(dim=0, index=index_gather).squeeze(dim=0)
        return action.detach().cpu().numpy()
