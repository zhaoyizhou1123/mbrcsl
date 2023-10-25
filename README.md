# MBRCSL
Model-based return-conditioned supervised learning

## Installation
1. Install dependencies according to `requirements.txt`

2. Add this repo directory to your `PYTHONPATH` environment variable.
```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```


## Run Experiments

### Point Maze
First, you can download the pointmaze dataset from this [Google drive link](https://drive.google.com/file/d/1y3B3er6k15bdjDb_9TyZRQi8Sv7gJFUM/view?usp=sharing). We recommend to store it under directory `../rl_dataset`. 


Then, you can test our main algorithm (MBRCSL) as well as other baselines in the paper by running

```[shell]
bash scripts/pointmaze/run.sh
```

Note: If you store the dataset in a customized directory, you need to modify the `data_dir` variable in the shell script accordingly.

To compare different output policies (MBCQL, MBRCSL-Gaussian), you need to run MBRCSL algorithm first to get the rollout dataset `rollout.dat`. By default, it will appear under a directory of the form `logs/pointmaze/mbrcsl/timestamp_[TIME]&[SEED]/rollout/checkpoint`, where `[TIME]` is the timestamp of the experiment (e.g., `23-1021-054021`), and `[SEED]` is an integer of the experiment seed. After that, you can test MBCQL and MBRCSL-Gaussian by running
```[shell]
bash scripts/pointmaze/ablation.sh
```
Remember to modify the command option `--rollout_ckpt_path` to the correct directory
that contains the rollout dataset. 


## Simulated robotics
You need to first install [roboverse](https://github.com/avisingh599/roboverse), which implemented the environments.

We use the same datasets as in [COG](https://github.com/avisingh599/cog). Please download the datasets from their [Google drive link](https://drive.google.com/drive/folders/1jxBQE1adsFT1sWsfatbhiZG6Zkf3EW0Q?usp=sharing). We recommend to store these datasets under directory `../rl_dataset`.

To test our main algorithm (MBRCSL) as well as baselines (except for MBCQL) in the paper, you can run
```[shell]
bash scripts/[TASK]/run.sh
```
Here `[TASK]` can be `pickplace`, `doubledraweropen` or `doubledrawercloseopen`, corresponding to tasks PickPlace, ClosedDrawer and BlockedDrawer in the paper, respectively. 

Note: If you store the datasets in a customized directory, you need to modify the `data_dir` variable in the shell script accordingly.

To test MBCQL algorithm, you need to run MBRCSL algorithm first to get the rollout dataset `rollout.dat`. By default, it will appear under a directory of the form `logs/[TASK]/mbrcsl/timestamp_[TIME]&[SEED]/rollout/checkpoint`, where `[TASK]` is the name of the task (`pickplace`, `doubledraweropen` or `doubledrawercloseopen`), `[TIME]` is the timestamp of the experiment (e.g., `23-1021-054021`), and `[SEED]` is an integer of the experiment seed. After that, you can test MBCQL by running
```[shell]
bash scripts/[TASK]/ablation.sh
```
Remember to modify the command option `--rollout_ckpt_path` to the correct directory
that contains the rollout dataset. 

## Acknowledgement
The framework of this repository, including part of the baseline algorithms (COMBO, MOPO, CQL) are built upon [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit). The DT and \%BC baselines are built upon [Decision Transformer](https://github.com/kzl/decision-transformer). 