# This script is used to schedule the execution of multiple experiments.
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 902 --fold 0
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 902 --fold 1
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 902 --fold 2
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 902 --fold 3
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 902 --fold 4