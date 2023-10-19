import gymnasium

import argparse
import numpy as np
import json
import os
import pickle

from .envs.point_maze import PointMaze
from .utils.maze_utils import set_map_cell

def create_env_dataset(args):
    '''
    Create env and dataset (if not created)
    '''
    maze_config = json.load(open(args.maze_config_file, 'r'))
    maze = maze_config["maze"]
    map = maze['map']  

    start = maze['start']
    goal = maze['goal']

    sample_args = maze_config["sample_args"]

    print(f"Create point maze")
    point_maze = PointMaze(data_path = os.path.join(args.data_dir, args.data_file), 
                        horizon = args.horizon,
                        maze_map = map,
                        start = np.array(start),
                        goal = np.array(goal),
                        sample_args = sample_args,
                        debug=False,
                        render=False)   
    env = point_maze.env_cls()
    trajs = point_maze.dataset[0]
    return env, trajs

def create_env(args):
    '''
    Create env(if not created)
    '''
    maze_config = json.load(open(args.maze_config_file, 'r'))
    maze = maze_config["maze"]
    map = maze['map']  

    start = maze['start']
    goal = maze['goal']

    target_map = set_map_cell(map, goal, 'g')
    target_map = set_map_cell(target_map, start, 'r')

    render_mode = "human" if args.render else "None"

    env = gymnasium.make('PointMaze_UMazeDense-v3', 
             maze_map = target_map, 
             continuing_task = False,
             max_episode_steps=args.horizon,
             render_mode=render_mode)
    
    return env

def load_dataset(data_path):
    '''
    Try to load dataset from daa_path. If fails, return none
    '''
    if data_path is not None and os.path.exists(data_path):
        with open(data_path, 'rb') as file:
            dataset = pickle.load(file) # Dataset may not be trajs, might contain other infos
        return dataset
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default=None,help='./dataset/maze2_smdacc_.dat')
    parser.add_argument('--horizon', type=int, default=300)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--maze_config_file', type=str, default='./config/maze2_simple_moredata.json')

    args = parser.parse_args()
    print(args)

    create_env(args)

