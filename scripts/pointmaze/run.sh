# ! Change it to your own directory
data_dir="../rl_dataset"

task=pointmaze
horizon=200

# mbrcsl
for seed in 0 1 2 3
do
python examples/pointmaze/run_mbrcsl_maze.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon} 
done

# combo
for seed in 0 1 2 3
do
python examples/pointmaze/run_combo_maze.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon} 
done

# mopo
for seed in 0 1 2 3
do
python examples/pointmaze/run_mopo_maze.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon} 
done

# cql
for seed in 0 1 2 3
do
python examples/pointmaze/run_cql_maze.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon} 
done

# dt
for seed in 0 1 2 3
do
python examples/pointmaze/run_dt_maze.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon} 
done

# %bc
for seed in 0 1 2 3
do
python examples/pointmaze/run_bc_maze.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon} 
done