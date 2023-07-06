import gymnasium as gym

import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from IPython.display import clear_output


class ESNetwork(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size=8,
        pop_size=5,
        target_score=500,
        learning_rate=1e-2,
        sigma=0.1,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

        self.target_score = target_score
        # controls noise rate
        self.sigma = sigma
        # learning rate
        self.lr = learning_rate
        self.score_tracking = deque(maxlen = 100)
        # "centroid" parameters
        self.master_weights = [param.data for param in self.parameters()]

        self.pop_size = pop_size
        self.populate()

        # pytorch has its autograd engine on by default - we won't use it so disable it
        map(lambda param: param.requires_grad_(False), self.parameters())
    
    def forward(self, x):
        return self.layers(x)
    
    def set_individual(self, individual_idx):
        """change which individual, and consequently parameter weights, are being used"""
        assert 0 <= individual_idx < self.pop_size
        for p_group, param in enumerate(self.parameters()):
            noise = self.sigma * self.population[individual_idx][p_group]
            # add noise to centroid
            param.data = self.master_weights[p_group] + noise

    def populate(self):
        """create new population with random weights"""
        # NOTE: we don't store PARAMETERS, but rather WEIGHTS
        self.population = []
        for _ in range(self.pop_size):
            idv_weights = []
            for params in self.parameters():
                # generate weights for this param group of this individual
                idv_weights.append(torch.randn(params.shape))
            self.population.append(idv_weights)
    
    def evolve(self, rewards):
        # avoid division by 0
        if torch.std(rewards) != 0:
            # on `unbiased` -- https://github.com/pytorch/pytorch/issues/1082
            normalized_rewards = (rewards - torch.mean(rewards)) / torch.std(rewards, unbiased=False)
            for p_group, param in enumerate(self.parameters()):
                    # group of parameters for each individual, has shape (population_size, *params[p_type].size)
                    A = torch.stack([individual[p_group] for individual in self.population])
                    '''
                    funny pytorch moment
                    we have A.shape = [pop_size, *param_size] and rewards.shape = [pop_size]
                    for the product to work we need to create singleton dimensions on rewards
                    problem being, some params are 2D, others are 1D, so we do this to generalize
                    '''
                    param_singletons = [1] * param.dim()
                    # scale each individual's weights by its performance
                    weighted_param_update = A * normalized_rewards.view(-1, *param_singletons)
                    # sum all individuals' weights - we now know what direction the master weights should go
                    weighted_param_update = weighted_param_update.sum(dim=0)
                    self.master_weights[p_group] += self.lr / (self.pop_size * self.sigma) * weighted_param_update

        # dynamic lr
        # https://github.com/BenSecret/Pytorch-Evolution-Strategies/blob/master/pytorch-evolution-strategies.py
        high_score = torch.max(rewards)
        self.score_tracking.append(high_score)
        self.learning_rate = (self.lr*5 + (self.target_score - np.mean(self.score_tracking))*0.000005)/6
        self.sigma = self.learning_rate * 10


class ESAgent:
    def __init__(
        self,
        env: gym.Env,
        hidden_layer_size=8,
        population_size=5,
        learning_rate=1e-2,
        sigma=0.1,
    ):
        """
        initialize an agent that uses the Evolutionary Systems algorithm

        :param env: the gym environment this agent will be trained/tested on
        :param hidden_layer_size: n. of nodes on the hidden layer
        :param population_size: how many individuals to create and run simultaneously
        :param learning_rate: learning rate for neural network updates
        :param sigma: how dispersed ES population's weights should be
        :type hidden_layer_size: int
        :type population_size: int
        :type learning_rate: int
        :type sigma: float
        """

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.env = env
        self.seed = seed
        self.pop_size = population_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        
        self.obs_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.hidden_size = hidden_layer_size

        self.init_nn()


        self.gen_scores = np.empty(self.pop_size)

        print('device:', self.device)

    def init_nn(self):
        self.nn = (
            ESNetwork(
                self.obs_size, 
                self.action_size, 
                self.hidden_size, 
                self.pop_size, 
                self.env.spec.reward_threshold,
                self.learning_rate,
                self.sigma
            )
            .to(self.device)
        )

    def choose_action(self, state):
        state = torch.from_numpy(state)
        selected_action = self.nn(state).argmax().item()
        return selected_action
    
    def take_step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        return next_state, reward, is_done
    
    def train_ep(self, individual):
        """train for a single episode"""
        ep_score = 0

        self.nn.set_individual(individual)

        state, _ = self.env.reset()
        while True:
            action = self.choose_action(state)
            next_state, reward, is_done = self.take_step(action)
            
            ep_score += reward

            if is_done:
                break
            state = next_state
        return ep_score
    
    def train(self, max_eps=None, plot=False, plotting_interval=10):
        """train for `max_eps` episodes"""

        # ma: mean average
        scores = []
        short_ma = deque(maxlen=5)
        long_ma = deque(maxlen=10)

        pop_idxs = range(self.pop_size)
        try:
            for ep in range(1, max_eps+1):
                # I think I made it very easy to make this paraleli.. parallelizable? word
                # I don't know how to create multiple gym envs though
                # with ThreadPool(processes=self.pop_size) as pool:
                #     scores = pool.map(train_ep, pop_idxs)
                # run the population batch
                ep_scores = [self.train_ep(idv) for idv in pop_idxs]
                ep_scores_tensor = torch.FloatTensor(ep_scores, device=self.device)

                self.nn.evolve(ep_scores_tensor)
                # create new random weight distribution
                self.nn.populate()
                
                if ep % 10 == 0:
                    # has it been doing worse in the short term than in the long(er) term?
                    # this prevents the "ES blunder"; all the recent population is bad,
                    # maybe we're just in a bad parameter space, so nuke it
                    if np.mean(short_ma) < np.mean(long_ma):
                        self.init_nn()
                        short_ma.clear()
                        long_ma.clear()

                scores.append(ep_scores)
                if plot and (ep % plotting_interval == 0):
                    self._plot(ep, scores)

                # if torch.mean(ep_scores_tensor) >= env.spec.reward_threshold:
                #     break

        except KeyboardInterrupt:
            self.save_state('saved-state-interrupt')
        else:
            self.save_state()
        
        self.env.close()

    def populate(self):
        self.population = np.empty(self.POPULATION_SIZE)
    
    def save_state(self, fname='saved-state'):
        state_dict = self.nn.state_dict()
        torch.save(state_dict, f'saved-states/{fname}.s{self.seed}.pt')
        print('saved state!')
        return state_dict

    def load_state(self, state_dict):
        self.nn.load_state_dict(state_dict)
        return state_dict
        
    def test(self):
        # TODO:
        raise NotImplementedError
  
    def _plot(self, ep, scores):
        # fig, (ax1,) = plt.subplots(1,1, figsize=(20, 5), num=1, clear=True)
        fig, ax1 = plt.subplots(1,1, num=1, clear=True)

        ax1.set_title(f'batch ep.: {ep} | total ep.: {ep*pop_size} | score: {np.mean(scores[-5:])}')
        ax1.set_ylabel('total reward')
        ax1.fill_between(range(len(scores)), np.max(scores, axis=-1), np.min(scores, axis=-1), alpha=0.5)
        ax1.plot(np.mean(scores, axis=-1))

        # clear_output(wait=True)
        ax1.clear()

        plt.show()

if __name__ == '__main__':
    plt.ion()

    # initialize gym
    env_id = "CartPole-v1"
    env = gym.make(env_id)

    # seeding
    seed = 14
    torch.manual_seed(seed)
    np.random.seed(seed)
    # in gym, we can seed by resetting the environment and giving an input seed
    env.reset(seed=seed)

    # parameters
    hidden_layer_size = 8
    pop_size = 5
    learning_rate = 1e-2
    sigma = 0.1
    agent = ESAgent(env, hidden_layer_size, pop_size, learning_rate, sigma)

    max_episodes = 30_000
    agent.train(max_episodes)