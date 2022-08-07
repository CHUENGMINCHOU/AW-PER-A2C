# AW-PER-A2C
The test code for the paper "Attention-based advantage actor-critic algorithm with prioritized experience replay for complex 2-D robotic motion planning".
The AW-PER-A2C is inspired by Crowd-Robot Interaction (https://arxiv.org/abs/1809.08835) and Self Imitation Learning (https://doi.org/10.48550/arXiv.1806.05635). The main part of our code is modified from codes in these two papers.

Abstract
Robotic motion planning in dense and dynamic indoor scenarios constantly challenges the researchers because of the motion
unpredictability of obstacles. Recent progress in reinforcement learning enables robots to better cope with the dense and
unpredictable obstacles by encoding complex features of the robot and obstacles into the encoders like the long-short term
memory (LSTM). Then these features are learned by the robot using reinforcement learning algorithms, such as the deep Q
network and asynchronous advantage actor critic algorithm. However, existing methods depend heavily on expert experiences
to enhance the convergence speed of the networks by initializing them via imitation learning. Moreover, those approaches
based on LSTM to encode the obstacle features are not always efficient and robust enough, therefore sometimes causing
the network overfitting in training. This paper focuses on the advantage actor critic algorithm and introduces an attention-
based actor critic algorithm with experience replay algorithm to improve the performance of existing algorithm from two
perspectives. First, LSTM encoder is replaced by a robust encoder attention weight to better interpret the complex features of
the robot and obstacles. Second, the robot learns from its past prioritized experiences to initialize the networks of the advantage
actor-critic algorithm. This is achieved by applying the prioritized experience replay method, which makes the best of past
useful experiences to improve the convergence speed. As results, the network based on our algorithm takes only around 15%
and 30% experiences to get rid of the early-stage training without the expert experiences in cases with five and ten obstacles,
respectively. Then it converges faster to a better reward with less experiences (near 45% and 65% of experiences in cases
with ten and five obstacles respectively) when comparing with the baseline LSTM-based advantage actor critic algorithm.
Our source code is freely available at the GitHub (https://github.com/CHUENGMINCHOU/AW-PER-A2C).

How to cite this paper:
Zhou, C., Huang, B., Hassan, H. et al. Attention-based advantage actor-critic algorithm with prioritized experience replay for complex 2-D robotic motion planning. J Intell Manuf (2022). https://doi.org/10.1007/s10845-022-01988-z



## Install library
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Package setup (in the file with setup.py) by pip3: 
        
        pip3 install -e .


## Run the test code
1. Visualize a test case with 5 obstacles
    First open the file AW-PER-A2C/ENVS/envs/configs/env.config and ensure "human_num = 5"
    Second open the file test.py and ensure the selected model is "aw-per-a2c-5obs.pkl"
    Third run the test code:
    
        python3 test.py --policy aw_per_a2c --output_dir ENVS/data/output --phase test --visualize --test_case 0

2. Visualize a test case with 10 obstacles
   The instruction to run test case with 10 obs is the same as that of 5 obs. Just modify relative codes to "human_num = 10" and "aw-per-a2c-10obs.pkl", and then run the test code.
