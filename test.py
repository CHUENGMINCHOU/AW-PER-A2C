import argparse
import torch
import logging
import configparser
import gym
import numpy as np
import os
from ENVS.envs.policy.policy_factory import policy_factory
from ENVS.envs.utils.robot import Robot
from ENVS.envs.policy.orca import ORCA

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='ENVS/envs/configs/env.config')
    parser.add_argument('--policy_config', type=str, default='ENVS/envs/configs/policy.config')
    parser.add_argument('--policy', type=str, default='aw_per_a2c')
    parser.add_argument('--output_dir', type=str, default='ENVS/data/output')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=True, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    #model_weights = os.path.join(args.output_dir, 'aw-per-a2c-5obs.pkl')
    model_weights = os.path.join(args.output_dir, 'aw-per-a2c-10obs.pkl')

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)

    model = policy.get_model()
    model.load_state_dict(torch.load(model_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)

    env = gym.make('sil-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)


    policy.set_phase(args.phase)
    policy.set_device(device)

    policy.set_env(env)
    policy.load_model(model)
    robot.print_info()
    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action, _= robot.act(ob)
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
