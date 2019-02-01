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
        self.fire_center = np.array([(self.forest_dimension+1)/2, (self.forest_dimension+1)/2])
        self.update_sim_every = 6

        self.feature_length = np.prod(self.image_dims) + 2 + 1 + 2
        self.action_length = 1
        self.reward_length = 1

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
            self.loss_fn = nn.MSELoss(reduction='elementwise_mean')


class UAV(object):
    healthy = 0
    on_fire = 1
    burnt = 2

    move_deltas = [(-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, -1), (0, -1), (1, -1)]  # excluded (0,0)
    fire_neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    def __init__(self, numeric_id=None, initial_position=None, fire_center=None, image_dims=(3, 3)):
        self.numeric_id = numeric_id
        self.initial_position = initial_position
        self.fire_center = fire_center
        self.image_dims = image_dims

        self.position = self.initial_position
        self.next_position = None

        self.image = None
        self.reached_fire = False
        self.rotation_vector = None
        self.closest_agent_id = None
        self.closest_agent_position = None
        self.closest_agent_vector = None

        self.features = None

        self.actions = []
        self.rewards = []

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
            self.rotation_vector = self.rotation_vector / norm

        d = [(np.linalg.norm(self.position-agent.position, 2), agent.numeric_id, agent.position)
             for agent in team.values() if agent.numeric_id != self.numeric_id]
        _, self.closest_agent_id, self.closest_agent_position = min(d, key=lambda x: x[0])

        self.closest_agent_vector = self.position - self.closest_agent_position
        norm = np.linalg.norm(self.closest_agent_vector)
        if norm != 0:
            self.closest_agent_vector = self.closest_agent_vector/norm

        return np.concatenate((self.image.ravel(), self.rotation_vector,
                              np.asarray(self.numeric_id > self.closest_agent_id)[np.newaxis],
                               self.closest_agent_vector))


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

    def __init__(self, mode='train', filename=None):
        self.mode = mode
        self.config = Config(config_type=self.mode)
        self.model = Policy(image_dims=self.config.image_dims).type(self.config.dtype)

        if mode == 'train':
            self.target = Policy(image_dims=self.config.image_dims).type(self.config.dtype)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

            self.sars = None
            self.eps = self.config.eps_ini
            self.reward_history = []
            self.loss_history = []
            self.num_train_episodes = 0

            self.print_enough_experiences = False

        if filename is not None:
            self.load_checkpoint(filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['state_dict'])

        if self.mode == 'train':
            self.target.load_state_dict(checkpoint['target_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.sars = checkpoint['replay']
            self.eps = checkpoint['epsilon']
            self.reward_history = checkpoint['reward_history']
            self.loss_history = checkpoint['plot_history']
            self.num_train_episodes = checkpoint[self.num_train_episodes]

    def save_checkpoint(self):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'target_dict': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'replay': self.sars,
            'epsilon': self.eps,
            'reward_history': self.reward_history,
            'loss_history': self.loss_history,
            'num_train_episodes': self.num_train_episodes
        }
        filename = 'madqn-' + time.strftime('%d-%b-%Y-%H%M') + '.pth.tar'
        torch.save(checkpoint, filename)

    def train(self, num_episodes):
        sim = LatticeForest(self.config.forest_dimension)
        number_agents = 10
        team = {i: UAV(numeric_id=i, fire_center=self.config.fire_center) for i in range(number_agents)}

        model_updates = 1

        for episode, seed in enumerate(range(num_episodes)):
            # for each simulation, set random seed and reset simulation
            np.random.seed(seed)
            sim.rng = seed
            sim.reset()

            # deploy agents randomly
            for agent in team.values():
                agent.position = np.random.choice(self.config.start, 2) + np.random.choice(self.config.perturb, 2)
                agent.initial_position = agent.position
                agent.reset()

            sim_updates = 1
            sim_control = defaultdict(lambda: (0.0, 0.0))

            while not sim.end and sim.iter < self.config.sim_iter_limit:
                forest_state = sim.dense_state()

                # determine action for each agent
                for agent in team.values():
                    agent.features = agent.update_features(forest_state, team)

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
                    agent.actions.append(action)
                    agent.next_position = np.asarray(actions2trajectory(agent.position, [action])[1])
                    agent.rewards.append(reward(forest_state, agent))

                    # determine treated trees based on agent movement
                    sim_control[xy2rc(sim.dims[0], agent.next_position)] = (0, self.config.delta_beta)

                # periodically update simulator
                if sim_updates % self.config.update_sim_every == 0:
                    sim.update(sim_control)
                    sim_control = defaultdict(lambda: (0, 0))
                    forest_state = sim.dense_state()

                sim_updates += 1

                # actually change agent position
                for agent in team.values():
                    agent.update_position()

                # do not update network if the simulation terminates early
                if sim.end:
                    continue

                for agent in team.values():
                    # skip agent if it has not reached the fire boundary
                    if not agent.reached_fire:
                        continue

                    # generate next features
                    next_features = agent.update_features(forest_state, team)

                    # add to experience set
                    data = np.zeros((1, 2*self.config.feature_length
                                     + self.config.action_length + self.config.reward_length))
                    data[0, 0:self.config.feature_length] = agent.features
                    data[0, self.config.feature_length] = agent.actions[-1]
                    reward_idx = self.config.feature_length + self.config.action_length
                    data[0, reward_idx] = agent.rewards[-1]
                    next_features_idx = self.config.feature_length+self.config.action_length+self.config.reward_length
                    data[0, next_features_idx:] = next_features

                    if self.sars is None:
                        self.sars = data
                    else:
                        self.sars = np.vstack((self.sars, data))

                # skip update if not enough experiences
                if self.sars is None or self.sars.shape[0] < self.config.min_experience_size or \
                    self.sars.shape[0] < self.config.batch_size:
                    continue
                elif not self.print_enough_experiences:
                    print('[MADQN] --- generated enough experiences')
                    self.print_enough_experiences = True

                # create mini-batch
                batch = self.sars[np.random.choice(self.sars.shape[0], self.config.batch_size, replace=False), :]
                batch_features = torch.from_numpy(batch[:, 0:self.config.feature_length])
                batch_features = Variable(batch_features).type(self.config.dtype)
                batch_actions = torch.from_numpy(batch[:, self.config.feature_length])
                batch_actions = Variable(batch_actions).type(torch.cuda.LongTensor)
                x = self.model(batch_features).gather(1, batch_actions.view(-1, 1)).squeeze()

                # calculate loss
                batch_rewards = batch[:, self.config.feature_length+self.config.reward_length]
                next_features_idx = self.config.feature_length+self.config.action_length+self.config.reward_length
                batch_next_features = Variable(torch.from_numpy(batch[:, next_features_idx:])).type(self.config.dtype)
                tt = self.target(batch_next_features).data.cpu().numpy()
                tt = batch_rewards + self.config.gamma*np.amax(tt, axis=1)
                tt = Variable(torch.from_numpy(tt), requires_grad=False).type(self.config.dtype)
                loss = self.config.loss_fn(x, tt)

                self.loss_history.append(loss.item())

                # back propagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # anneal exploration rate
                if self.eps > self.config.eps_fin:
                    self.eps += -(self.config.eps_ini - self.config.eps_fin)/self.config.anneal_range
                    if self.eps <= self.config.eps_fin:
                        print('[MADQN] --- finished annealing exploration rate')

                # update target network periodically
                if model_updates % self.config.update_target_every == 0:
                    self.target = copy.deepcopy(self.model)
                    print('[MADQN] --- updated target network (%d)' % (model_updates/self.config.update_target_every))
                model_updates += 1

                # drop from memory if too many elements
                if self.sars.shape[0] > self.config.memory_size:
                    self.sars = self.sars[self.sars.shape[0]-self.config.memory_size, :]

            seed_reward = np.mean([np.mean(agent.rewards) for agent in team.values()])
            fraction_healthy = sim.stats[0]/np.sum(sim.stats)
            self.reward_history.append(seed_reward)
            print('[MADQN] seed %03d: %03d sim iterations, %0.4f average agent reward, %0.4f healthy '
                  % (seed, sim.iter, seed_reward, fraction_healthy))
            self.num_train_episodes += 1

        # save results at end of training
        self.save_checkpoint()

    def test(self, capacity=None):
        pass
