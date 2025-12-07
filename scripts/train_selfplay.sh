#!/bin/sh
env="MyAirCombatV1"
scenario="1v1/MyConfig/myaircombat_v1"
algo="ppo"
exp="v1"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=1 python train/train_my_combat.py --env-name "MyAirCombatV1" --algorithm-name "ppo" --scenario-name "1v1/MyConfig/myaircombat_v1" --experiment-name "v1" --seed 1 --n-training-threads 1 --n-rollout-threads 1 --log-interval 1 --save-interval 1 --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 --num-mini-batch 4 --buffer-size 4096 --num-env-steps 1e8 --lr 1e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 0.01 --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 --user-name "rqs" --use-wandb --wandb-name "2473395984-shanghaijiaotong"

python train/train_my_combat.py --env-name "MyAirCombat" --algorithm-name "ppo" --scenario-name "1v1/MyConfig/myaircombat" --experiment-name "v1" --seed 1 --n-training-threads 1 --n-rollout-threads 1 --log-interval 1 --save-interval 1 --selfplay-algorithm "fsp" --n-choose-opponents 1 --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8

#首先安装环境配置 conda env create -f environment.yml
#然后激活环境 conda activate jsbsim
#然后运行下方训练命令行
#wandb api key 03493940ae311b1517976549b9bd093e6784bd42
python train/train_my_combat.py --env-name "MyAirCombatV1" --algorithm-name "ppo" --scenario-name "1v1/MyConfig/myaircombat_v1" --experiment-name "v5" --seed 1 --n-training-threads 1 --n-rollout-threads 1 --log-interval 1 --save-interval 1 --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 --num-mini-batch 8 --buffer-size 32768 --num-env-steps 1e8 --lr 3e-4 --gamma 0.997 --ppo-epoch 10 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 0.01 --use-gae --gae-lambda 0.95 --use-proper-time-limits --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 --user-name "rqs" --use-wandb --wandb-name "2473395984-shanghaijiaotong"
