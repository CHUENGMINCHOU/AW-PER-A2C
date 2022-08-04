
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
    def __init__(self, input_dim, self_state_dim, lstm_hidden_dim):#6/50
        super(Net, self).__init__()

        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

        # action network
        self.act_fc1 = nn.Linear(self_state_dim + lstm_hidden_dim, 512)
        #self.act_fc2 = nn.Linear(128, 128)
        #self.act_fc3 = nn.Linear(256, 256)
        self.act_fc4 = nn.Linear(512, 81)
        self.act_fc4.weight.data.mul_(0.1)

        # value network
        self.value_fc1 = nn.Linear(self_state_dim + lstm_hidden_dim, 512)
        #self.value_fc2 = nn.Linear(128, 128)
        #self.value_fc3 = nn.Linear(256, 256)
        self.value_fc4 = nn.Linear(512, 1) 
        self.value_fc4.weight.data.mul(0.1)

    def forward(self, state):
        #joint state from lstm
         
        size = state.shape
        dim = state.dim()
        if dim == 2:
            state = state.unsqueeze(0)
            size = state.shape
      
        self_state = state[:, 0, :self.self_state_dim]
        #human_state = state[:, self.self_state_dim:]
        #print('human_state', human_state.shape)
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(state, (h0, c0))
        #print('output', output)
        hn = hn.squeeze(0)
        #print('self_state', self_state.shape)
        joint_state = torch.cat([self_state, hn], dim=1)
        #print('joint_state', joint_state)

        # action
        act = self.act_fc1(joint_state)
        act = F.relu(act)
        #act = self.act_fc2(act)
        #act = F.relu(act)
        #act = self.act_fc3(act)
        #act = F.relu(act)
        pi = Categorical(F.softmax(self.act_fc4(act), dim=-1))

        # value
        v = self.value_fc1(joint_state)
        v = F.relu(v)
        #v = self.value_fc2(v)
        #v = F.relu(v)
        #v = self.value_fc3(v)
        #v = F.relu(v)
        v = self.value_fc4(v)


        #print('v', v, 'pi', pi)
        return v, pi

class LstmRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'LSTM-RL'
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

    def configure(self, config):
        self.set_common_parameters(config)
        #mlp_dims = [int(x) for x in config.get('lstm_rl', 'mlp2_dims').split(', ')]
        global_state_dim = config.getint('lstm_rl', 'global_state_dim')
        self.with_om = config.getboolean('lstm_rl', 'with_om')
        with_interaction_module = config.getboolean('lstm_rl', 'with_interaction_module')

        self.multiagent_training = config.getboolean('lstm_rl', 'multiagent_training')
        self.model = Net(self.input_dim(), self.self_state_dim, global_state_dim)
        self.action_space = build_action_space()

        logging.info('Policy: {}LSTM-RL {} pairwise interaction module'.format(
            'OM-' if self.with_om else '', 'w/' if with_interaction_module else 'w/o'))

    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        #self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def load_model(self, model):
        self.model = model

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

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)



    def predict(self, state):
        def dist(human):
            # sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)

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



