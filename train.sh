# # tmux 2
# CUDA_VISIBLE_DEVICES=2 \
# python train.py \
# --exp-id rel-pos-cur \
# --seed 1 \
# --dataset interaction-dataset \
# --dataset-path ./datasets/interaction_dataset/data/multi_agent \
# --use-map-lanes \
# --model-type Autobot-Joint \
# --num-modes 6 \
# --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --tx-hidden-size 384 --dropout 0.1 \
# --entropy-weight 0.0 --kl-weight 1.0 --use-FDEADE-aux-loss \
# --batch-size 32 --learning-rate 0.0005 --learning-rate-sched 150 --num-epochs 150

# tmux 3
CUDA_VISIBLE_DEVICES=3 \
python train.py \
--exp-id man-soc \
--seed 1 \
--dataset interaction-dataset \
--dataset-path ./datasets/interaction_dataset/data/multi_agent \
--use-map-lanes \
--model-type Autobot-Joint \
--num-modes 6 \
--hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --tx-hidden-size 384 --dropout 0.1 \
--entropy-weight 0.0 --kl-weight 1.0 --use-FDEADE-aux-loss \
--batch-size 32 --learning-rate 0.0005 --learning-rate-sched 150 --num-epochs 150
