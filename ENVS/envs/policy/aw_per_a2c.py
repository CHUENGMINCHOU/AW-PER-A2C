from torch.nn.functional import softmax
import logging
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
    speeds = [(np.exp((i + 1) / 5) - 1) / (np.e - 1) * 1 for i in range(5)]
    rotations = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    action_space = [ActionXY(0, 0)]
    for rotation, speed in itertools.product(rotations, speeds):
        action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
    
    return action_space


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net




class Net(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        #self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

        #layers for action and value
        self.act_fc1 = nn.Linear(mlp3_input_dim, 128)
        #self.act_fc2 = nn.Linear(256, 64)
        self.act_fc3 = nn.Linear(128, 81)
        self.act_fc3.weight.data.mul_(0.1)

        self.value_fc1 = nn.Linear(mlp3_input_dim, 128)
        #self.value_fc2 = nn.Linear(256, 64)
        self.value_fc3 = nn.Linear(128, 1) 
        self.value_fc3.weight.data.mul(0.1)


    def forward(self, state):
        size = state.shape
        dim = state.dim()
        if dim == 2:
            state = state.unsqueeze(0)
            size = state.shape

        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx

        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        #print('joint_state:', joint_state.shape)

        # action
        #print('joint_state', joint_state.shape)
        act = self.act_fc1(joint_state)
        act = F.relu(act)
        #act = self.act_fc2(act)
        #act = F.relu(act)
        pi = Categorical(F.softmax(self.act_fc3(act), dim=-1))
        # value
        v = self.value_fc1(joint_state)
        v = F.relu(v)
        #v = self.value_fc2(v)
        #v = F.relu(v)
        v = self.value_fc3(v)

        return v, pi


class AW_PER_A2C(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'AW_PER_A2C'
        self.kinematics = 'holonomic'
        self.multiagent_training = 'False'
        self.with_interaction_module = None
        self.interaction_module_dims = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        #mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = Net(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        self.action_space = build_action_space()

        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def load_model(self, model):
        self.model = model

    def get_attention_weights(self):
        return self.model.attention_weights

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
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')

        if self.reach_destination(state):
            action = ActionXY(0, 0)
            action_indice = self.find_action_indice(action, self.action_space)
            return action, action_indice

        states_tensor = self.transform(state)
        #print('states_tensor', states_tensor)
        _, pi = self.model(states_tensor)#.data.item()
        action_indice = self.select_action(pi) # the action is the indice in action space
        action_indice = action_indice.item()
        action = self.action_space[action_indice]

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return action, action_indice



    def select_action(self, pi):
        return pi.sample()


    def transform(self, state):
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
        return state_tensor

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


    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()










