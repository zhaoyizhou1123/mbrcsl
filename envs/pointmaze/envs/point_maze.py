import gymnasium

from ..envs.base import BaseOfflineEnv
from ..utils.maze_utils import set_map_cell
from ..samplers.maze_sampler import MazeSampler

class PointMaze(BaseOfflineEnv):
    def __init__(self, data_path, horizon, maze_map, start, goal, 
                 sample_args,
                 debug=False, render=False):
        '''
        data_path: path to dataset
        maze_map: list(list), basic map of the game, only specifies 0, 1. R and G are specified by start and goal.
        start, array (2,), int, the start point of the game we want to learn
        goal, array (2,), int, the goal point of the game we want to learn
        horizon: horizon of every trajectory
        n_trajs: number of trajectories for dataset
        sample_starts: list, list of starts for sampling
        sample_goals: list, list of goals for sampling
        '''

        self.MAZE_MAP = maze_map
        target_map = set_map_cell(self.MAZE_MAP, goal, 'g')
        target_map = set_map_cell(target_map, start, 'r')

        render_mode = "human" if render else None
        env_cls = lambda : gymnasium.make('PointMaze_UMazeDense-v3', 
                                    maze_map = target_map, 
                                    continuing_task = False,
                                    max_episode_steps=self.horizon,
                                    render_mode=render_mode)
        
        sampler = MazeSampler(horizon=horizon,
                              maze_map=self.MAZE_MAP,
                              target_start=start,
                              target_goal=goal,
                              debug=debug,
                              render=render)
        
        super().__init__(data_path, env_cls, horizon,
                         sampler = sampler, sample_args=sample_args)
