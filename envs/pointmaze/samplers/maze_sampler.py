'''
Implement data behaviorr for point maze.
The sampling maze is similar to the true maze, but has different starting state and goals
'''

from copy import deepcopy
from typing import Union, List, Dict
import gymnasium
import numpy as np

from ..samplers.base import BaseSampler
from ..policies.maze_expert import WaypointController
from ..utils.maze_utils import VALID_VALUE, set_map_cell, cell2xy, terminated
from ..utils.trajectory import Trajectory


'''
All elements are list type.
- 'observations': obs['observation']. 
- 'actions'
- 'rewards'
- 'returns': the return achieved, for RCSL methods
- 'timesteps': the current timestep
- 'terminated': True if achieved goal
- 'truncated': True if reached time limit
- 'infos': the info output by env
'''

class MazeSampler(BaseSampler):
    def __init__(self, horizon, maze_map, target_start, target_goal, debug = False, render = False) -> None:
        '''
        horizon: int, sampling horizon
        maze_map: list(list), map of the game, only specifies 0, 1. R and G are specified by start and goal.
        target_start, array (2,), int, the start point of the game we want to learn. Used for recording data
        target_goal, array (2,), int, the goal point of the game we want to learn
        debug, bool. If True, print debugging info
        '''
        super().__init__()
        self.horizon = horizon
        self.MAZE_MAP = deepcopy(maze_map) # The basic map, without c,r,g. Cannot be changed

        self.target_start = target_start
        self.target_goal = target_goal

        self.debug = debug
        self.render = render
        
        self.VEL_THRESHOLD = 0.5 # velocity threshold for goal reaching

        if self.debug:
            print(f"MazeSampler: Target goal {self.target_goal}")

    def collect_trajectories(self, sample_args: dict):
        '''
        Sample Multiple trajectories.
        sample_args: dict, contain keys:
        - starts: list(np.array (2,) | List | List(list))
        - goals: list(np.array (2,) | List)

        Return: (list(Trajectory), horizon, map, target_start, target_goal). Elements 3-5 are for env
        '''
        assert 'starts' in sample_args and 'goals' in sample_args, f"sample_args is expected to have keys 'starts' and 'goals' "
        starts = sample_args['starts']
        goals = sample_args['goals']
        repeats = sample_args['repeats']
        randoms = sample_args['randoms']
        assert len(starts) == len(goals), f"collect_trajectories: starts and goals are expected to have the same length!"


        trajs_ = []
        for idx in range(len(starts)):
            start = starts[idx]
            goal = goals[idx]
            repeat = repeats[idx]
            random_end = randoms[idx]
            trajectorys = self._collect_single_traj(start, goal, repeat, random_end)
            trajs_ += trajectorys

        return (trajs_, self.horizon, self.MAZE_MAP, self.target_start, self.target_goal)

        
    def _collect_single_traj(self, start, goals: Union[List[List],List,List[np.ndarray]], repeat, random_end: bool):
        '''
        Collect one trajectory.
        start: np.array (2,), type=int. Initial (row,col)
        goal: np.array (2,), type=int. Goal (row,col). If list, then multiple goals, reach them one by one
        repeat: int. times to repeat
        random_end: If True, do random walk when goal reached
        Return: Trajectory
        '''
        # Set up behavior environment
        if self.render:
            render_mode = 'human'
        else:
            render_mode = None

        # Configure data maze and env, the env to record data
        data_map = set_map_cell(self.MAZE_MAP, start, 'r')
        data_map = set_map_cell(data_map, self.target_goal, 'g')

        data_env = gymnasium.make('PointMaze_UMazeDense-v3', 
                              maze_map= data_map, 
                              continuing_task = False,
                              max_episode_steps=self.horizon,
                              render_mode = render_mode)
        
        # Set up controller, only the map matters, start and goal is unimportant
        controller = WaypointController(maze = deepcopy(data_env.maze))
        
        # Initialize data to record
        trajs = []
        for _ in range(repeat):
            observations_ = []
            actions_ = []
            rewards_ = []
            achieved_rets_ = [] # The total reward that has achieved, used to compute rtg
            timesteps_ = []
            terminateds_ = []
            truncateds_ = []
            infos_ = []

            # Copy goal list
            if type(goals[0]) != np.ndarray and type(goals[0]) != list: # single goal element
                list_goals = [goals]
            else:
                list_goals = deepcopy(goals)
            if self.debug:
                print(f"Goals id: {list_goals}")

            # reset, data_env, behavior_env only differ in reward
            seed = np.random.randint(0, 1000)
            obs, _ = data_env.reset(seed=seed)
            if self.debug:
                print(f"True goal: {obs['desired_goal']}")
            cur_goal = list_goals.pop(0) # maintain the current goal of controller
            cur_goal_xy = cell2xy(self.MAZE_MAP, cur_goal) # Convert to coordinate
            if self.debug:
                print(f"Current goal xy: {cur_goal_xy}")
            # Initialize return accumulator, terminated, truncated, info
            achieved_ret = 0
            data_terminated = False
            truncated = False
            info = None
            goals_reached = False # maintain whether all goals are reached
            is_last_goal = (len(list_goals) == 0) # maintain whether this is the last goal

            for n_step in range(self.horizon):
                observations_.append(deepcopy(obs['observation']))
                achieved_rets_.append(deepcopy(achieved_ret))
                timesteps_.append(deepcopy(n_step))
                terminateds_.append(data_terminated) # We assume starting point is unfinished
                truncateds_.append(truncated)
                infos_.append(info)

                if goals_reached: # All goals were reached
                    if random_end:
                        action = data_env.action_space.sample()
                    else: # Towards the target goal
                        controller_obs = deepcopy(obs)
                        controller_obs['desired_goal'] = cur_goal_xy
                        action = controller.compute_action(controller_obs)
                else:
                    # Whether current goal reached. Omit vel threshold if it is last goal and random_end=True
                    if terminated(obs, cur_goal_xy, vel_threshold=self.VEL_THRESHOLD, omit_vel=is_last_goal and random_end): 
                        if len(list_goals) > 0: # Still have other goals, update current goal
                            cur_goal = list_goals.pop(0)
                            cur_goal_xy = cell2xy(self.MAZE_MAP, cur_goal)
                            is_last_goal = (len(list_goals)==0) # Update whether it is last goal
                            controller_obs = deepcopy(obs)
                            if self.debug:
                                print(f"Changing current goal xy to {cur_goal_xy}")
                            controller_obs['desired_goal'] = cur_goal_xy
                            action = controller.compute_action(controller_obs)
                        else: # All goals are reached, turn to status 'goal_reached'
                            if self.debug:
                                print("Goal reached")
                            goals_reached = True
                            if random_end:
                                action = data_env.action_space.sample()
                            else: # Towards the target goal
                                controller_obs = deepcopy(obs)
                                controller_obs['desired_goal'] = cur_goal_xy
                                action = controller.compute_action(controller_obs)
                    else: # Current goal not reached, remain chasing current goals
                        controller_obs = deepcopy(obs)
                        controller_obs['desired_goal'] = cur_goal_xy
                        action = controller.compute_action(controller_obs)

                obs, reward, data_terminated, truncated, info = data_env.step(action)
                if self.debug:
                    print(f"Step {n_step}, data maze, current pos {obs['observation']}, terminated {data_terminated}, reward {reward}")

                actions_.append(deepcopy(action))
                rewards_.append(deepcopy(reward))

                # Update return
                achieved_ret += reward

            # Compute returns. Note that achieved_ret is now total return
            total_ret = achieved_ret
            if self.debug:
                print(f"Total return: {total_ret}\n -----------------")
            returns_ = [total_ret - achieved for achieved in achieved_rets_]
            trajs.append(Trajectory(observations = observations_, 
                          actions = actions_, 
                          rewards = rewards_, 
                          returns = returns_, 
                          timesteps = timesteps_, 
                          terminated = terminateds_, 
                          truncated = truncateds_, 
                          infos = infos_))
        # behavior_env.close()
        data_env.close()
        return trajs
    
    def get_expert_return(self, repeat=10):
        '''
        Collect one trajectory.
        start: np.array (2,), type=int. Initial (row,col)
        goal: np.array (2,), type=int. Goal (row,col)
        repeat: int. times to repeat
        random_end: If True, do random walk when goal reached
        Return: Trajectory
        '''

        # Configure behavior maze map, the map for controller to take action
        behavior_map = set_map_cell(self.MAZE_MAP, self.target_start, 'r')
        behavior_map = set_map_cell(behavior_map, self.target_goal, 'g')

        # Set up behavior environment
        if self.render:
            render_mode = 'human'
        else:
            render_mode = None
        # render_mode = None

        # print(f"Behavior_env render mode {render_mode}")
        behavior_env = gymnasium.make('PointMaze_UMazeDense-v3', 
                              maze_map= behavior_map, 
                              continuing_task = False,
                              max_episode_steps=self.horizon,
                              render_mode = render_mode)

        # Set up controller
        controller = WaypointController(maze = deepcopy(behavior_env.maze))
        
        # if self.debug:
        #     print(f"behavior_env==data_env: {behavior_env==data_env}")
        
        # Initialize data to record
        rets = []
        for epoch in range(repeat):

            # reset, data_env, behavior_env only differ in reward
            seed = np.random.randint(0, 1000)
            behavior_obs, _ = behavior_env.reset(seed=seed)
            if self.debug:
                print(f"Goal: {behavior_obs['desired_goal']}")

            # Initialize return accumulator, terminated, truncated, info
            achieved_ret = 0

            for n_step in range(self.horizon):

                # if data_terminated: # Reached true goal, don't move, dummy action, change reward to 1
                #     # action = np.zeros(2,)
                #     # reward = 1

                #     # Continue control
                #     pass
                    
                # else: 
                    # controller uses the 'desired_goal' key of obs to know the goal, not the goal mark on the map
                action = controller.compute_action(behavior_obs)
                # action = controller.compute_action(behavior_obs)

                behavior_obs, reward, behavior_terminated, _, _ = behavior_env.step(action)
                if self.debug:
                    print(f"Step {n_step}, behavior maze, current pos {behavior_obs['achieved_goal']}, terminated {behavior_terminated}")

                # Update return
                achieved_ret += reward

            # Compute returns. Note that achieved_ret is now total return
            print(f"Epoch {epoch}, total return {achieved_ret}")
            rets.append(achieved_ret)

        behavior_env.close()

            # if data_terminated:
            #     print(f"Warning: data_env already reached goal, quit sampling immediately")
            #     break
            # if behavior_terminated:
            #     print(f"Behavior env finished")
            #     break
        return sum(rets) / len(rets)



