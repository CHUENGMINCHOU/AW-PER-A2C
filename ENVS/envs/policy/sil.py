from ENVS.envs.policy.policy import Policy
import numpy as np
from ENVS.envs.utils.utils import *
import torch
import torch.nn as nn
import itertools
from torch.distributions.categorical import Categorical
from ENVS.envs.utils.state import ObservableState, FullState
from ENVS.envs.utils.action import *
import torch.nn.functional as F

def build_action_space():
    """
    Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
    """
    #holonomic = True if self.kinematics == 'holonomic' else False
    speeds = [(np.exp((i + 1) / 5) - 1) / (np.e - 1) * 1 for i in range(5)]
    #if holonomic:
    #    rotations = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rotations = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    #else:
    #    rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

    action_space = [ActionXY(0, 0)]
    for rotation, speed in itertools.product(rotations, speeds):
        #if holonomic:
        #    action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
        action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
        #else:
        #    action_space.append(ActionRot(speed, rotation))
    
    return action_space


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # action network
        self.act_fc1 = nn.Linear(13, 128)
        self.act_fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, 81)
        self.mu.weight.data.mul_(0.1)
        # torch.log(std)
        #self.logstd = nn.Parameter(torch.zeros(17))

        # value network
        self.value_fc1 = nn.Linear(13, 128)
        self.value_fc2 = nn.Linear(128, 128)
        self.value_fc3 = nn.Linear(128, 1)
        self.value_fc3.weight.data.mul(0.1)

    def forward(self, x):
        # action
        act = self.act_fc1(x)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        #mean = self.mu(act)  # N, num_actions
        pi = Categorical(F.softmax(self.mu(act), dim=-1))
        #logstd = self.logstd.expand_as(mean)
        #std = torch.exp(logstd)
        #action = torch.normal(mean, std)

        # value
        v = self.value_fc1(x)
        v = F.tanh(v)
        v = self.value_fc2(v)
        v = F.tanh(v)
        v = self.value_fc3(v)

        # action prob on log scale
        #logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, pi#, logprob, mean


class SIL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'SIL'
        self.kinematics = 'holonomic'
        self.multiagent_training = 'False'

    def configure(self, config):
        self.model = Net()
        #self.model_actor = Actor()
        #self.model_critic = Critic()
        self.action_space = build_action_space()
        self.gamma = config.getfloat('rl', 'gamma')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def load_model(self, model):
        self.model = model

    #def load_model_critic(self, model_critic):
    #    self.model_critic = model_critic

    def find_action_indice(self, actions, possible_actions):
        possible_actions = np.array(possible_actions)
        actions = torch.tensor(actions).unsqueeze(0)
        indices = np.zeros((actions.shape[0], ), dtype=np.int)

        actions_x = actions[:,0]
        actions_y = actions[:,1]
        possible_actions_x = possible_actions[:,0]
        possible_actions_y = possible_actions[:,1]
        diff_x = actions_x[:,np.newaxis] - possible_actions_x
        diff_y = actions_y[:,np.newaxis] - possible_actions_y
        dist_sq = diff_x ** 2 + diff_y ** 2
        indices = np.argmin(dist_sq, axis=1)
        indices = torch.tensor(indices).squeeze(0)
        return indices


    def predict(self, state):
        if self.reach_destination(state):
            action = ActionXY(0, 0)
            action_indice = self.find_action_indice(action, self.action_space)
            return action, action_indice

        state_ = self.transform(state)
        #pi = self.model_actor(state_)
        _, pi = self.model(state_)
        action_indice = self.select_action(pi) # the action is the indice in action space
        action_indice = action_indice.item()
        action = self.action_space[action_indice]

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return action, action_indice


    def select_action(self, pi):
        return pi.sample()

    def transform(self, state):
        assert len(state.human_states) == 1
        
        state = torch.Tensor(state.self_state + state.human_states[0])
        state = state.unsqueeze(0)
        state = self.rotate(state).to(self.device)
        return state

    def rotate(self, state):

        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]

        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)

        return new_state
 
