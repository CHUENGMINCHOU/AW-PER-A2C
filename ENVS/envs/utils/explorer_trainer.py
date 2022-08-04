import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ENVS.envs.utils.info import *


def average(input_list):
    if input_list:
        return sum(np.array(input_list).tolist()) / len(input_list)
    else:
        return 0


class Explorer(object): # run k epis to update buffer in training; to test the model in eva/testing
    def __init__(self, args, env, robot, device, memory, capacity, gamma, policy, model, batch_size, train_batches):
        self.env = env
        self.robot = robot
        self.device = device
        self.gamma = gamma
        self.robot_policy = policy
        self.model = model
        #self.model_critic = critic
        self.args = args
        self.buffer = memory
        self.capacity = capacity
        self.batch_size = batch_size
        self.trainer = Trainer(self.model, self.buffer, self.device, self.batch_size, self.robot_policy)
        self.train_batches = train_batches

        self.total_steps = []
        self.total_rewards = []
        self.optimizer_ac = optim.Adam(self.model.parameters(), lr=3e-6)

    def compute_returns(self, rewards, masks):
        R = 0
        returns = []
        for i in reversed(range(len(rewards))):
            gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
            R = rewards[i] + gamma_bar * R * masks[i]
            returns.insert(0, R)
        return returns

    def update_buffer(self, trajectory, avg_rewards, episode):
        positive_reward = False
        
        for (state, a, r) in trajectory:
            if r > 0:
                positive_reward = True
                break

        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.total_steps) > self.capacity and len(self.total_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        for (state, action_indice, R) in trajectory:
            state = state.squeeze(0).detach().numpy()
            self.buffer.add(state, action_indice, R)

    def _update_network(self, states, actions_indice, returns, ada_ent):
        states = torch.tensor(states)
        actions_indice = torch.tensor(actions_indice)
        returns = torch.tensor(returns)

        self.optimizer_ac.zero_grad()
        #print('states', states.shape)
        states = states.to(self.device)
        values, pi= self.model(states.float())
        action_log_probs, dist_entropy = self.evaluate_actions(pi, actions_indice)

        advantages = returns - values

        action_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()
        total_loss = action_loss + 0.5 * value_loss - ada_ent * dist_entropy

        total_loss.backward()
        self.optimizer_ac.step()

    def evaluate_actions(self, pi, actions_indice):
        return pi.log_prob(actions_indice), pi.entropy().mean()

    def adaptive_entropy(self, episode):
        #if episode < 20001:
        #    ada_ent = 0.001 - episode * (0.001 - 0.0001)/20000
        #else:
        #    ada_ent = 0.0001

        # fixed learning rate
        ada_ent = 1e-6

        return ada_ent

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False,  episode=None, print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []

        ada_ent = self.adaptive_entropy(episode)

        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            states_np = []
            actions = []
            actions_indice = []
            rewards = []
            dones = []
            masks = []

            while not done:
                action, action_indice = self.robot.act(ob)
                ob_next, reward, done, info = self.env.step(action)

                state = self.robot.policy.last_state
                states.append(state)

                state_np = np.array(state)
                states_np.append(state_np)

                actions_indice.append(action_indice)
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
                dones.append(done)
                masks.append(torch.tensor([1-done], dtype=torch.float32, device=self.device))
                ob = ob_next

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            returns = self.compute_returns(rewards, masks)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            avg_cumulative_rewards = average(cumulative_rewards)
            if update_memory:
            #if episode <= self.num_epi_per:
                trajectory = []
                for (state, action_indice, return_) in list(zip(states, actions_indice, returns)):
                    trajectory.append([state, action_indice, return_])
                self.update_buffer(trajectory, avg_cumulative_rewards, episode)
            
            if self.args.train_a2c:
                self._update_network(states_np, actions_indice, returns, ada_ent)

            if self.args.train_sil:
            #if episode <= self.num_epi_per:
                self.trainer.optimize_batch(self.train_batches)


        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))

        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))




###################
# trainer for sil
###################
class Trainer(object):
    def __init__(self, model, memory, device, batch_size, policy):
        self.network = model
        self.buffer = memory
        self.device = device
        self.batch_size = batch_size
        self.policy = policy
        # some other parameters...
        self.optimizer_sil = optim.Adam(self.network.parameters(), lr=1e-4)
        self.max_nlogp = 5.0
        self.mini_batch_size = 64
        self.clip = 1
        self.entropy_coef = 1e-5
        #self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.sil_beta = 0.1
        self.w_value = 0.01
    
    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            batch_size_ = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size_, beta=self.sil_beta)
        else:
            return None, None, None, None, None

    def evaluate_actions_sil(self, pi, actions_indice):
        return pi.log_prob(actions_indice.squeeze(-1)).unsqueeze(-1), pi.entropy().unsqueeze(-1)

    def optimize_batch(self, train_batches):
        for n in range(train_batches):
            states, actions_indice, returns, weights, idxes = self.sample_batch(self.batch_size)
            #mean_adv, num_valid_samples = 0, 0
            if states is not None:
                # need to get the masks (the positive advantages).
                states = torch.tensor(states, dtype=torch.float32)
                actions_indice = torch.tensor(actions_indice, dtype=torch.float32).unsqueeze(1)
                returns = torch.tensor(returns, dtype=torch.float32)#.unsqueeze(1)
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
                max_nlogp = torch.tensor(np.ones((len(idxes), 1)) * self.max_nlogp, dtype=torch.float32)

                # start to next...
                states = states.squeeze(0)
                #print('states', states.shape)
                states = states.to(self.device)
                value, pi = self.network(states)
                action_log_probs, dist_entropy = self.evaluate_actions_sil(pi, actions_indice)

                action_log_probs = -action_log_probs
                clipped_nlogp = torch.min(action_log_probs, max_nlogp)

                advantages = returns - value.squeeze(1)
                advantages = advantages.detach()
                masks = (advantages.cpu().numpy() > 0).astype(np.float32)
                # get the num of vaild samples
                num_valid_samples = np.sum(masks)
                num_samples = np.max([num_valid_samples, self.mini_batch_size])
                # process the mask
                masks = torch.tensor(masks, dtype=torch.float32)

                # clip the advantages...
                clipped_advantages = torch.clamp(advantages, 0, self.clip)
                # start to get the action loss...
                action_loss = torch.sum(clipped_advantages * weights * clipped_nlogp) / num_samples
                entropy_reg = torch.sum(weights * dist_entropy * masks) / num_samples
                policy_loss = action_loss - entropy_reg * self.entropy_coef

                # start to process the value loss..
                # get the value loss
                delta = torch.clamp(value - returns, -self.clip, 0) * masks
                delta = delta.detach()
                value_loss = torch.sum(weights * value * delta) / num_samples
                total_loss = policy_loss + 0.5 * self.w_value * value_loss

                self.optimizer_sil.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer_sil.step()

                # update the priorities
                self.buffer.update_priorities(idxes, clipped_advantages.cpu().numpy())





    
