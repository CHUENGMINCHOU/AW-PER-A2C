from ENVS.envs.policy.linear import Linear
from ENVS.envs.policy.orca import ORCA
from ENVS.envs.policy.sil import SIL
from ENVS.envs.policy.lstm_rl import LstmRL
from ENVS.envs.policy.aw_per_a2c import AW_PER_A2C


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['lstm_rl'] = LstmRL
policy_factory['sil'] = SIL
policy_factory['aw_per_a2c'] = AW_PER_A2C
policy_factory['none'] = none_policy
