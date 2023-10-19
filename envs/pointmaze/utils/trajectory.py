from collections import namedtuple
from typing import List, Optional, Union, Tuple, Dict
import numpy as np

Trajectory = namedtuple(
    "Trajectory", ["observations", "actions", "rewards", "returns", "timesteps", "terminated", "truncated", "infos"])
'''
Each attribute is also a list, element shape according to env, length equal to horizon
'''

def show_trajectory(traj: Trajectory, timesteps = None):
    '''
    print a trajectory for specified timesteps
    timesteps: None or list
    '''
    traj_finish = False
    # idx = 0

    obss = traj.observations
    acts = traj.actions
    rs = traj.rewards
    rets = traj.returns
    ts = traj.timesteps
    terminateds = traj.terminated
    truncateds = traj.truncated
    infos = traj.infos

    if timesteps is None:
        loop_range = range(len(obss))
    else:
        loop_range = timesteps
    
    for idx in loop_range:
        assert idx in range(len(obss)), f"idx {idx} out of range {len(obss)}!"
        obs = obss[idx]
        act = acts[idx]
        r = rs[idx]
        ret = rets[idx]
        t = ts[idx]
        terminated = terminateds[idx]
        truncated = truncateds[idx]
        info = infos[idx]
        print(f"Timestep {t}, obs {obs}, act {act}, r {r}, ret {ret}")
        if terminated:
            print(f"Terminated")
        if truncated:
            print(f"Truncated")

def get_pointmaze_dataset(
        trajs: List,
        sample_ratio: float = 1.) -> Tuple[Dict, np.ndarray, float]:
    '''
    Return:
    dataset: Dict. key 'rtgs' is set to zero, it will not be used in training
    init_obss
    max offline return
    '''
    num_trajs = int(len(trajs) * sample_ratio)
    idxs = np.random.choice(len(trajs), size=(num_trajs), replace = False)
    valid_trajs = [trajs[i] for i in list(idxs)]

    obss = [traj.observations[0:-1] for traj in valid_trajs]
    next_obss = [traj.observations[1:] for traj in valid_trajs]
    acts = [traj.actions[0:-1] for traj in valid_trajs]
    rs = [traj.rewards[0:-1] for traj in valid_trajs]
    init_obss = [traj.observations[0:1] for traj in valid_trajs] # initial observations

    obss = np.concatenate(obss, axis=0)
    next_obss = np.concatenate(next_obss, axis=0)
    acts = np.concatenate(acts, axis=0)
    rs = np.concatenate(rs, axis=0)
    terminals = np.array([False]).repeat(obss.shape[0])
    weights = np.ones_like(rs).astype(np.float32)
    init_obss = np.concatenate(init_obss, axis=0)

    rets = [sum(traj.rewards) for traj in valid_trajs]
    rtgs = np.zeros_like(rs) 

    dataset = {
        "observations": obss,
        "next_observations": next_obss,
        "actions": acts,
        "rewards": rs,
        "rtgs": rtgs,
        "terminals": terminals,
        "weights": weights}

    return dataset, init_obss, max(rets)

if __name__ == '__main__':
    traj = Trajectory(observations=0,
                      actions=1,
                      rewards=2,
                      returns=3,
                      timesteps=4,
                      terminated=5,
                      truncated=6,
                      infos=7)
    traj = traj._asdict()

    from typing import Optional, Union, Tuple, Dict
    def f(traj: Dict[str, int]):
        print(traj['observations'])
    f(traj)
