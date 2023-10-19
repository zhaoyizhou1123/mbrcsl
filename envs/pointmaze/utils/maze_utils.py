from copy import deepcopy
from typing import List, Union, Tuple, Dict
import numpy as np
from gymnasium import Wrapper

VALID_VALUE = [0,1,'c','g','r'] # valid value of cell

def set_map_cell(MAP, pos, value):
    '''
    Return a map with cell [pos] of MAP set to [value], MAP is not changed.
    MAP: list(list), the basic map
    pos: np.array (2,), type=int, the (row,col) of the cell. Upperleft cell is (0,0)
    value: {0,1,'c','g','r'}, the value to set the cell

    Output: list(list), a modified map
    '''
    assert value in VALID_VALUE, f"_set_map_cell: Invalid value {value}!"
    new_map = deepcopy(MAP)
    new_map[pos[0]][pos[1]] = value

    return new_map

def cell2xy(MAP: List[List], pos: Union[Tuple, List, np.ndarray], noise_bound = 0.25) -> np.ndarray:
    '''
    Convert cell (row-col) discription to x-y coordinate with noise
    '''
    pos_y, pos_x = pos[0], pos[1] # row is y coordinate, col is x x coordinate
    max_pos_x = len(MAP[0]) - 1 # max x cell position
    max_pos_y = len(MAP) - 1 # max y cell position
    center_x = max_pos_x / 2 # center cell position x, int.5 type
    center_y = max_pos_y / 2 # center cell position y, int.5 type
    assert 0 <= pos_x and pos_x <= max_pos_x, f"Invalid x position {pos_x}"
    assert 0 <= pos_y and pos_y <= max_pos_y, f"Invalid y position {pos_y}"
    coordinate_x = pos_x - center_x + np.random.uniform(low = -noise_bound, high = noise_bound)
    coordinate_y = -pos_y + center_y + np.random.uniform(low = -noise_bound, high = noise_bound) # Large col index means small y coordinate
    return np.array([coordinate_x, coordinate_y])

def terminated(obs: Union[Dict, np.ndarray], desired_goal: np.ndarray, pos_threshold = 0.5, vel_threshold = 0.1, omit_vel=False) -> bool:
    '''
    Test whether the goal is reached.
    obs: Dict (full observation) | np.ndarray (pos+velocity obs)
    omit_vel: If True, omit vel_threshold
    '''
    if type(obs) == dict:
        obs = obs['observation']
    cur_pos = obs[0:2] # np.ndarray
    dist = np.linalg.norm(cur_pos - desired_goal)
    # print(f"Distance to current goal: {dist}")
    if omit_vel:
        return dist <= pos_threshold
    else:
        return dist <= pos_threshold and abs(obs[2]) <= vel_threshold and abs(obs[3]) <= vel_threshold

class PointMazeObsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['observation']

    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return obs['observation']
    
    def step(self, action):
        '''
        use truncated signal as terminal
        '''
        next_obs, reward, _, truncated, info = self.env.step(action)
        next_obs = self.observation(next_obs)
        return next_obs, reward, truncated, info

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return self.observation(obs)