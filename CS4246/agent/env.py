import gym
from gym.utils import seeding
from gym_grid_driving.envs.grid_driving import LaneSpec, DenseReward, GridDrivingEnv, SparseReward


def construct_random_lane_env(observation_type='tensor', rewards=None):
    config = {'observation_type': observation_type, 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=7, speed_range=[-3, -1]),
                        LaneSpec(cars=8, speed_range=[-3, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=7, speed_range=[-3, -1]), 
                        LaneSpec(cars=8, speed_range=[-3, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=7, speed_range=[-3, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=8, speed_range=[-3, -1])],
              'random_lane_speed': True,
            }
    if rewards:
        config['rewards'] = rewards
    return gym.make('GridDriving-v0', **config)


class MDPEnv(GridDrivingEnv):

    def __init__(self, **kwargs):
        kwargs['rewards'] = DenseReward
        super(MDPEnv, self).__init__(**kwargs)

