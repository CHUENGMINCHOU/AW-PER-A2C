# AW-PER-A2C
The test code for the paper "Attention-based advantage actor-critic algorithm with prioritized experience replay for complex 2-D robotic motion planning".
The AW-PER-A2C is inspired by Crowd-Robot Interaction (https://arxiv.org/abs/1809.08835) and Self Imitation Learning (https://doi.org/10.48550/arXiv.1806.05635). The main part of our code is modified from codes in these two papers.


## Install library
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Package setup (in the file with setup.py) by pip3: pip3 install -e .


## Run the test code
1. Visualize a test case with 5 obstacles
    First open the file AW-PER-A2C/ENVS/envs/configs/env.config and ensure "human_num = 5"
    Second open the file test.py and ensure the selected model is "aw-per-a2c-5obs.pkl"
    Third run the test code:
    
        python3 test.py --policy aw_per_a2c --output_dir ENVS/data/output --phase test --visualize --test_case 0

2. Visualize a test case with 10 obstacles
   The instruction to run test case with 10 obs is the same as that of 5 obs. Just modify relative codes to "human_num = 10" and "aw-per-a2c-10obs.pkl", and then run the test code.
