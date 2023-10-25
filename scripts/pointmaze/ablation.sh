# ! Change the `rollout_ckpt_path` to your own directory containg "rollout.dat" file

for seed in 0 1 2 3
do
    python examples/pointmaze/run_mbcql_maze.py --rollout_ckpt_path logs/pointmaze/mbrcsl/rollout_s${seed} --seed ${seed}
done
wait

for seed in 0 1 2 3
do
    python examples/pointmaze/run_rcsl_gauss_maze.py --rollout_ckpt_path logs/pointmaze/mbrcsl/rollout_s${seed} --seed ${seed}
done
