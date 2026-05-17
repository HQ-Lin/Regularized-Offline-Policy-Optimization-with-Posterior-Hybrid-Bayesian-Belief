__credits__ = ["Rushiv Arora"]

import numpy as np

from gym import utils
from gym import spaces
import gym

from builtins import range
import numpy as np
from scipy.integrate import odeint


class CurrencyExchange(gym.Env):
    def __init__(self):
        self.obs_low = np.array([0., 0., 0.])
        self.obs_high = np.array([50., 100., 5.])
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        self.price_mu = 1.5
        self.price_sigma = 0.2
        self.price_theta = 0.05
        self.init_price_mu = 1.0
        self.init_price_sigma = 0.05
        self.num_steps = 20
        self.dt = 1
        self.state = self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, a):
        t, m, p = self.state
        t_next = t + 1
        
        exchange_ratio = np.clip(a[0], 0, 1)
        amount_to_exchange = m * exchange_ratio
        m_next = m - amount_to_exchange
        
        reward = amount_to_exchange * p
        
        p_next = p + self.price_theta * (self.price_mu - p) * self.dt + \
                 self.price_sigma * np.random.normal() * np.sqrt(self.dt)
        p_next = np.clip(p_next, 0, 5)

        terminal = bool(t_next >= self.num_steps or m_next < 0.1)

        s_next = np.array([t_next, m_next, p_next])
        self.state = s_next.copy()
        return s_next, reward, terminal, {}

    def reset(self):
        t = 0
        m = 100.
        p = np.random.normal(loc=self.init_price_mu, scale=self.init_price_sigma)
        s = np.array([t, m, p])
        self.state = s.copy()
        return s
