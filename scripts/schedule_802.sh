# This script is used to schedule the execution of multiple experiments.
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 802 --fold 0 -y
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 802 --fold 1 -y
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 802 --fold 2 -y
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 802 --fold 3 -y
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 802 --fold 4 -y