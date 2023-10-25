# Task specific hyperparameters
task=doubledraweropen
horizon=50

for seed in 0 1 2 3
do
# ! Change it to your own directory containing "rollout.dat" file
ckpt_path="logs/${task}/mbrcsl/timestamp_23-1018-s${seed}_final/rollout/checkpoint" 

python examples/roboverse/run_mbcql_roboverse.py \
    --seed ${seed} \
    --rollout_ckpt_path ${ckpt_path} \
    --task ${task} \
    --horizon ${horizon} 
done