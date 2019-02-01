import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from collections import defaultdict
import copy
import numpy as np
import os
import sys
import time

sys.path.append(os.getcwd() + '/simulators')
from fires.LatticeForest import LatticeForest

from rlUtilities import *


class Config(object):

    def __init__(self, config_type='train'):
        self.dtype = torch.cuda.FloatTensor

        # simulator parameters
        self.forest_dimension = 50
        self.delta_beta = 0.15/0.2763
        self.image_dims = (3, 3)
        self.fire_center = (self.forest_dimension+1)/2
        self.update_sim_every = 6

        if config_type == 'train':
            self.save_directory = '/checkpoints'

            # simulation iteration cutoff
            self.sim_iter_limit = 100

            # agent initialization locations
            self.start = np.arange(self.forest_dimension//3//2, self.forest_dimension, self.forest_dimension//3)
            self.perturb = np.arange(-self.forest_dimension//3//2 + 1, self.forest_dimension//3//2 + 1, 1)

            # replay memory
            self.memory_size = 1000000
            self.min_experience_size = 5000

            # target network instance
            self.update_target_every = 6000  # 6000

            # optimizer
            self.gamma = 0.95
            self.batch_size = 32
            self.learning_rate = 1e-4

            # exploration
            self.eps_ini = 1
            self.eps_fin = 0.15
            self.anneal_range = 20000  # 40000

            # loss function
            self.loss_fn = nn.MSELoss(size_average=True)


class UAV(object):
    healthy = 0
    on_fire = 1
    burnt = 2

    move_deltas = [(-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, -1), (0, -1), (1, -1)]  # excluded (0,0)
    fire_neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    def __init__(self, numeric_id=None, position=None, fire_center=None, image_dims=(3, 3)):
        self.numeric_id = numeric_id
        self.position = position
        self.fire_center = fire_center
        self.image_dims = image_dims

        self.next_position = None

        self.image = None
        self.reached_fire = False
        self.rotation_vector = None
        self.closest_agent_id = None
        self.closest_agent_position = None
        self.closest_agent_vector = None

        self.features = None
        self.action = None
        self.reward = None

    def reset(self):
        self.reached_fire = False
        self.next_position = None

    def update_position(self):
        self.position = self.next_position
        self.next_position = None

    def update_features(self, forest_state, team):
        height, width = forest_state.shape
        self.image = latticeforest_image(forest_state, xy2rc(height, self.position), self.image_dims)
        image_center = (self.image_dims[0]-1)//2, (self.image_dims[1]-1)//2
        if self.image[image_center[0], image_center[1]] in [self.on_fire, self.burnt]:
            self.reached_fire = True

        self.rotation_vector = self.position - self.fire_center
        norm = np.linalg.norm(self.rotation_vector, 2)
        if norm != 0:
            self.rotation_vector /= norm

        d = [(np.linalg.norm(self.position-agent.position, 2), agent.numeric_id, agent.position)
             for agent in team.values() if agent.numeric_id != self.numeric_id]
        _, self.closest_agent_id, self.closest_agent_position = min(d, key=lambda x: x[0])

        self.closest_agent_vector = self.position - self.closest_agent_position
        norm = np.linalg.norm(self.closest_agent_vector)
        if norm != 0:
            self.closest_agent_vector /= norm

        return np.concatenate(self.image.ravel(), self.position, self.rotation_vector,
                              np.asarray(self.numeric_id > self.closest_agent_id), self.closest_agent_vector)


class Policy(nn.Module):

    def __init__(self, image_dims=(3, 3)):
        super(Policy, self).__init__()

        # inputs: image + rotation vector + id compare + relative agent vector
        self.network = nn.Sequential(
            nn.Linear(np.prod(image_dims) + 2 + 1 + 2, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 9)
        )

    def forward(self, features):
        return self.network(features)


class MADQN(object):
    healthy = 0
    on_fire = 1
    burnt = 2

    def __init__(self, mode='train', filename=None):
        self.mode = mode
        self.config = Config(config_type=self.mode)
        self.model = Policy(image_dims=self.config.image_dims).type(self.config.dtype)

        if mode == 'train':
            self.target = Policy(image_dims=self.config.image_dims).type(self.config.dtype)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

            self.sars = None
            self.eps = self.config.eps_ini
            self.reward_hist = []
            self.loss_hist = []
            self.num_train_episodes = 0

        if filename is not None:
            self.load_checkpoint(filename)
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['state_dict'])

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['state_dict'])

        if self.mode == 'train':
            self.target.load_state_dict(checkpoint['target_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.sars = checkpoint['replay']
            self.eps = checkpoint['epsilon']
            self.reward_hist = checkpoint['plot_reward']
            self.loss_hist = checkpoint['plot_loss']
            self.num_train_episodes = checkpoint[self.num_train_episodes]

    def save_checkpoint(self):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'target_dict': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'replay': self.sars,
            'epsilon': self.eps,
            'plot_reward': self.reward_hist,
            'plot_loss': self.loss_hist,
            'num_train_episodes': self.num_train_episodes
        }
        filename = 'madqn-' + time.strftime('%d-%b-%Y-%H%M') + '.pth.tar'
        torch.save(checkpoint, filename)

    def train(self, num_episodes=110):
        sim = LatticeForest(self.config.forest_dimension)
        number_agents = 10
        team = {i: UAV(numeric_id=i, fire_center=self.config.fire_center) for i in range(number_agents)}

        for episode, seed in enumerate(range(num_episodes)):
            # for each simulation, set random seed and reset simulation
            np.random.seed(seed)
            sim.rng = seed
            sim.reset()

            # deploy agents randomly
            for agent in team.values():
                agent.position = np.random.choice(self.config.start, 2) + np.random.choice(self.config.perturb, 2)
                agent.reset()

            target_updates = 1
            sim_updates = 1
            sim_control = defaultdict(lambda: (0, 0))

            while not sim.end and sim.iter < self.config.sim_iter_limit:

                # determine action for each agent
                for agent in team.values():
                    agent.features = agent.update_features(sim.dense_state(), team)

                    action = None
                    if not agent.reached_fire:
                        # move to fire center
                        action = move_toward_center(agent)

                    else:
                        # use heuristic for exploration
                        if np.random.rand() <= self.eps:
                            action = heuristic(agent)

                        # otherwise use network
                        else:
                            q_features = Variable(torch.from_numpy(agent.features)).type(self.config.dtype)
                            q_values = self.model(q_features.unsqueeze(0))[0].data.cpu().numpy()
                            action = np.argmax(q_values)

                    # get reward
                    agent.action = action
                    agent.next_position = np.asarray(actions2trajectory(agent.position, [action])[1])
                    agent.reward = reward(agent)

                    # determine treated trees based on action
                    sim_control = to-do

                # periodically update simulator
                if sim_updates % self.config.update_sim_every == 0:
                    sim.update(sim_control)
                    sim_control = defaultdict(lambda: (0, 0))

                sim_updates += 1

                # actually change agent position
                for agent in team.values():
                    agent.update_position()

                # do not update network if the simulation terminates early
                if sim.end:
                    continue

                for agent in team.values():
                    if not agent.reached_fire:
                        continue

                    next_features = agent.features(sim.dense_state, team)

                if self.sars is None or self.sars.shape[0] < self.config.min_experience_size or \
                    self.sars.shape[0] < self.config.batch_size:
                    continue

                loss = 0
                batch = sars[np.random.choice(self.sars.shape[0], self.config.batch_size, replace=False), :]
                batch_states =

                if self.eps > self.config.eps_fin:
                    self.eps += -(self.config.eps_ini - self.config.eps_fin)/self.config.anneal_range

                if target_updates % self.config.update_target_every == 0:
                    self.target = copy.deepcopy(self.model)
                target_updates += 1

                if self.sars.shape[0] > self.config.memory_size:
                    self.sars = self.sars[self.sars.shape[0]-self.config.memory_size, :]


    def test(self):
        pass
