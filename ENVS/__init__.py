from gym.envs.registration import register

register(
    id='sil-v0',
    entry_point='ENVS.envs:CrowdSim',
)
