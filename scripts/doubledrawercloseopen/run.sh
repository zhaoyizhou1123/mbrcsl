# ! Change it to your own directory
data_dir="../rl_dataset"

# Task specific hyperparameters
task=doubledrawercloseopen
horizon=80

# mbrcsl
for seed in 0 1 2 3
do
python examples/roboverse/run_mbrcsl_roboverse.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon} \
    --behavior_epoch 40
done

# cql
for seed in 0 1 2 3
do
python examples/roboverse/run_cql_roboverse.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon} \
    --cql-weight 2.0
done

# combo
for seed in 0 1 2 3
do
python examples/roboverse/run_combo_roboverse.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon}
done

# dt
for seed in 0 1 2 3
do
python examples/roboverse/run_dt_roboverse.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon}
done

# bc (with diffusion policy)
for seed in 0 1 2 3
do
python examples/roboverse/run_diffusionbc_roboverse.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon}
done
